use anyhow::{Context as _, Result};
use cloud_llm_client::predict_edits_v3::Excerpt;
use collections::HashMap;
use edit_prediction_context::{EditPredictionExcerpt, Line};
use gpui::http_client::{Method, Request};
use gpui::{AppContext, AsyncApp, Entity, http_client::HttpClient};
use indoc::indoc;
use language::{Anchor, Buffer, BufferSnapshot, OffsetRangeExt as _, Point};
use open_ai::{FunctionDefinition, MessageContent};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use smol::io::AsyncReadExt;
use std::path::PathBuf;
use std::{
    collections::VecDeque,
    fmt::Write,
    path::Path,
    sync::{Arc, LazyLock},
};

use crate::Event;
use crate::assemble_excerpts::assemble_excerpts;
use crate::retrieval_search::run_retrieval_searches;
use cloud_zeta2_prompt::write_codeblock;

/// Search for relevant code
///
/// Alaways run all queries at once with a single invocation of this tool.
#[derive(Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchToolInput {
    /// An array of queries to run in parallel for gathering context
    #[schemars(length(max = 5))]
    pub queries: Box<[SearchToolQuery]>,
}

/// Search for relevant code by path and their content
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Hash)]
pub struct SearchToolQuery {
    /// A glob pattern to match file paths in the codebase to search in.
    pub glob: String,
    /// A regular expression to match code contents within the matched files.
    pub regex: String,
}

pub static TOOL_SCHEMA: LazyLock<(serde_json::Value, String)> = LazyLock::new(|| {
    let schema = language_model::tool_schema::root_schema_for::<SearchToolInput>(
        language_model::LanguageModelToolSchemaFormat::JsonSchemaSubset,
    );

    let description = schema
        .get("description")
        .and_then(|description| description.as_str())
        .unwrap()
        .to_string();

    (schema.into(), description)
});

pub struct JumpLocation {
    pub buffer: Entity<Buffer>,
    pub anchor: Anchor,
}

#[derive(Serialize)]
struct OpenRouterWrapper {
    #[serde(flatten)]
    request: open_ai::Request,
    provider: OpenRouterProvider,
}

#[derive(Serialize)]
pub struct OpenRouterProvider {
    only: Option<Vec<String>>,
}

pub async fn predict_jump(
    active_full_path: Arc<Path>,
    cursor_position: Point,
    events: VecDeque<Event>,
    project: Entity<Project>,
    http_client: Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<Option<JumpLocation>> {
    eprintln!("\n\nRequesting jump");

    // todo!
    let events = cx.update(|cx| {
        events
            .into_iter()
            .filter_map(|event| event.to_request_event(cx))
            .collect::<Vec<_>>()
    })?;

    let search_queries = cx.background_spawn({
        let http_client = http_client.clone();
        let active_full_path = active_full_path.clone();
        async move {
            let prompt = build_jump_prompt(&active_full_path, cursor_position, &events);
            eprintln!("Jump prompt:\n{prompt}");

            let (tool_schema, tool_description) = TOOL_SCHEMA.clone();

            let request_body = OpenRouterWrapper {
                request: open_ai::Request {
                    // model: "qwen3:8b".into(),
                    model: "qwen/qwen3-coder-30b-a3b-instruct".into(),
                    messages: vec![open_ai::RequestMessage::User {
                        content: open_ai::MessageContent::Plain(prompt),
                    }],
                    stream: false,
                    max_completion_tokens: None,
                    stop: Default::default(),
                    temperature: 0.7,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    tools: vec![open_ai::ToolDefinition::Function {
                        function: FunctionDefinition {
                            name: cloud_zeta2_prompt::retrieval_prompt::TOOL_NAME.to_string(),
                            description: Some(tool_description),
                            parameters: Some(tool_schema),
                        },
                    }],
                    prompt_cache_key: None,
                    reasoning_effort: None,
                },
                provider: OpenRouterProvider {
                    only: Some(vec!["nebius/fp8".into()]),
                },
            };

            let request = Request::builder()
                .method(Method::POST)
                // .uri("http://localhost:11434/v1/chat/completions")
                .uri("https://openrouter.ai/api/v1/chat/completions")
                .header(
                    "Authorization",
                    format!("Bearer {}", std::env::var("OPENROUTER_API_KEY").unwrap()),
                )
                .header("Content-Type", "application/json")
                .header("HTTP-Referer", "https://zed.dev")
                .header("X-Title", "Zed Editor")
                .body(serde_json::to_string(&request_body)?.into())?;

            let mut response = http_client.send(request).await?;
            let mut buf = Vec::new();
            response.body_mut().read_to_end(&mut buf).await?;

            if !response.status().is_success() {
                anyhow::bail!("Jump request failed: {}", String::from_utf8_lossy(&buf));
            }

            let response: open_ai::Response = serde_json::from_slice(&buf)?;
            dbg!(&response);

            anyhow::Ok((request_body, response))
        }
    });

    let (mut request_body, mut response) = search_queries.await?;

    let choice = response
        .choices
        .pop()
        .context("No choices in jump response")?;
    let open_ai::RequestMessage::Assistant {
        content: _,
        tool_calls,
    } = &choice.message
    else {
        anyhow::bail!("Jump response didn't include an assistant message");
    };

    let mut queries: Vec<cloud_zeta2_prompt::retrieval_prompt::SearchToolQuery> = Vec::new();
    let mut tool_call_id = None;

    for tool_call in tool_calls {
        tool_call_id.get_or_insert(tool_call.id.clone());
        let open_ai::ToolCallContent::Function { function } = &tool_call.content;
        if function.name != cloud_zeta2_prompt::retrieval_prompt::TOOL_NAME {
            log::warn!(
                "Jump response tried to call an unknown tool: {}",
                function.name
            );

            continue;
        }

        let input: SearchToolInput = serde_json::from_str(&function.arguments)
            .with_context(|| format!("invalid search json {}", &function.arguments))?;
        queries.extend(input.queries.into_iter().map(|query| {
            cloud_zeta2_prompt::retrieval_prompt::SearchToolQuery {
                glob: query.glob,
                syntax_node: vec![],
                content: Some(query.regex),
            }
        }));
    }

    let Some(tool_call_id) = tool_call_id else {
        anyhow::bail!("No searches in jump response");
    };

    if queries.is_empty() {
        anyhow::bail!("No queries in jump response");
    }

    let results = run_retrieval_searches(
        queries,
        project.clone(),
        #[cfg(feature = "eval-support")]
        None,
        cx,
    )
    .await?;
    dbg!(&results);

    if results.is_empty() {
        return anyhow::Ok(None);
    }

    // todo! move to background

    let mut combined_results = String::new();
    let mut result_buffers = HashMap::default();

    for (buffer, ranges) in results {
        let (snapshot, full_path) = buffer.read_with(cx, |buffer, cx| {
            (
                buffer.snapshot(),
                buffer
                    .file()
                    .map(|file| file.full_path(cx))
                    .unwrap_or_else(|| PathBuf::from("untitled")),
            )
        })?;

        let ranges = ranges
            .into_iter()
            .map(|range| {
                let point_range = range.to_point(&snapshot);
                Line(point_range.start.row)..Line(point_range.end.row)
            })
            .collect::<Vec<_>>();

        let excerpts = assemble_excerpts(&snapshot, ranges);

        write_codeblock(
            &full_path,
            &excerpts,
            &[],
            Line(snapshot.max_point().row),
            true,
            &mut combined_results,
        );

        result_buffers.insert(full_path.clone(), (buffer, snapshot));
    }
    eprintln!("{combined_results}");

    request_body.request.tools.clear();
    request_body.request.messages.extend([
            choice.message,
            open_ai::RequestMessage::Tool {
                content: MessageContent::Plain(combined_results),
                tool_call_id,
            },
            open_ai::RequestMessage::User {
                content: MessageContent::Plain(format!("{JUMP_INSTRUCTIONS}\nAssume that no more edits are required in the current file ({}). Only suggest jumping to other files.", active_full_path.display())),
            },
        ]);

    let request = Request::builder()
        .method(Method::POST)
        // .uri("http://localhost:11434/v1/chat/completions")
        .uri("https://openrouter.ai/api/v1/chat/completions")
        .header(
            "Authorization",
            format!("Bearer {}", std::env::var("OPENROUTER_API_KEY").unwrap()),
        )
        .header("Content-Type", "application/json")
        .header("HTTP-Referer", "https://zed.dev")
        .header("X-Title", "Zed Editor")
        .body(serde_json::to_string(&request_body)?.into())?;

    let mut response = http_client.send(request).await?;
    let mut buf = Vec::new();
    response.body_mut().read_to_end(&mut buf).await?;
    dbg!(String::from_utf8_lossy(&buf));

    if !response.status().is_success() {
        anyhow::bail!("Jump request failed: {}", String::from_utf8_lossy(&buf));
    }

    let mut response: open_ai::Response = serde_json::from_slice(&buf)?;

    if response.choices.is_empty() {
        return anyhow::Ok(None);
    }

    let choice = response
        .choices
        .pop()
        .context("No choices in jump response")?;

    let open_ai::RequestMessage::Assistant {
        content: Some(MessageContent::Plain(response)),
        tool_calls: _,
    } = &choice.message
    else {
        anyhow::bail!("Jump response didn't include an assistant message");
    };

    dbg!(response);

    let (file_path, line) = response
        .trim()
        .split_once("```jump")
        .context("Missing open fence")?
        .1
        .split_once("```")
        .context("Missing closing fence")?
        .0
        .trim()
        .split_once(":")
        .context("Invalid jump response")?;

    dbg!(file_path, line);

    let line = line.parse::<u32>()?;

    let (buffer, snapshot) = result_buffers
        .get(Path::new(file_path))
        .context("File not found in search results")?;

    anyhow::Ok(Some(JumpLocation {
        buffer: buffer.clone(),
        anchor: snapshot.anchor_after(Point::new(line.saturating_sub(1), 0)),
    }))
}

pub fn build_jump_prompt(
    active_full_path: &Path,
    cursor_position: Point,
    events: &[cloud_llm_client::predict_edits_v3::Event],
) -> String {
    let mut events_str = String::new();

    for event in events {
        write!(&mut events_str, "```diff\n{event}```\n\n").unwrap();
    }

    let events_str = events_str.trim_end_matches("\n\n");

    SEARCH_INSTRUCTIONS
        .to_string()
        .replace(
            "{CURSOR_PATH}",
            active_full_path.display().to_string().as_str(),
        )
        .replace("{CURSOR_LINE}", &(cursor_position.row + 1).to_string())
        .replace("{EDIT_HISTORY}", events_str)
}

const SEARCH_INSTRUCTIONS: &str = indoc! {r#"
    You are part of an edit prediction system in a code editor.

    The user has made a series of changes, and your role is to predict a single far-away location
    that needs to be edited as a result.

    ## Cursor location

    The cursor is currently located at `{CURSOR_PATH}:{CURSOR_LINE}`.

    Assume all necessary changes near this location are or will be done, and focus on changes that
    are needed in other parts of the codebase.

    ## Edit history

    Carefully analyze the edit history in order to infer what the user is currently trying to accomplish,
    and gather facts about the changes they have made.

    {EDIT_HISTORY}

    Use the `search` tool to find more information about the changes and potential locations for the next edit.

    ### When you find changes to usages

    - Did they pass a new argument to a function whose declaration hasn't been updated yet?
    - Did they use a method on a type/class that hasn't been added yet?
    - Did they use a method from a class/interface/trait that hasn't been implemented/derived on the type yet?
    - Did they start using a package or library that hasn't been added to the project yet?

    If the code suggets the item in question already existed, but is now being used in a different way,
    search for its declaration in order to determine whether changes are necessary.

    Alternatively, if the changes suggest the item is newly used, you should perform two parallel searches:
        1. Search for the declaration of item to see whether it already exists and whether its definition needs to be updated.
        2. Search for the class/type/module/configruation where it _should_ be defined, so that you can suggest jumping
           to it if needs to be added.

    ### When you find changes to declarations

    - Did they change the definition of a type/class/table by adding, removing, or altering fields?
    - Did they add an argument to a function?
    - Did they split a function into multiple functions? Or merge multiple functions into one?
    - Did they change the type of a field or function argument?
    - Did they move a field from one type to another?

    In these cases, you should search for usages of the affected item, so that you can see their current state
    and suggest jumping to them if necessary.

    If the affected item is public, make sure to include other files that may reference it in your search.
    If the name of the affected item is unique enough, search for it in the entire project.
"#};

const JUMP_INSTRUCTIONS: &str = indoc! {"
    Now analyze the search results, and explain your findings in 1 or 2 sentences.

    If no more edits are needed, output `None`.

    If another edit is needed, output the target file path and line number, like this:

    ```jump
    {project_name}/path/to/file.rs:123
    ```
"};
