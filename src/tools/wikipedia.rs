use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Wikipedia tool for searching and fetching Wikipedia articles
pub struct WikipediaTool;

#[derive(Debug, Deserialize)]
struct WikipediaInput {
    query: String,
    action: Option<String>,
    limit: Option<u32>,
    language: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WikipediaSearchResult {
    title: String,
    snippet: String,
    wordcount: Option<u32>,
}

#[derive(Debug, Serialize)]
struct WikipediaResponse {
    action: String,
    query: String,
    language: String,
    results: Vec<WikipediaSearchResult>,
    summary: Option<String>,
}

#[async_trait]
impl Tool for WikipediaTool {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn description(&self) -> &str {
        "Search Wikipedia articles and get article summaries. Supports multiple languages and can either search for articles or get detailed summaries of specific pages."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or article title"
                },
                "action": {
                    "type": "string",
                    "enum": ["search", "summary"],
                    "description": "Action to perform: 'search' to find articles, 'summary' to get article content (default: search)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of search results to return (default: 5, max: 20)"
                },
                "language": {
                    "type": "string",
                    "description": "Wikipedia language code (default: en). Examples: en, es, fr, de, it, pt, ru, ja, zh"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let params: WikipediaInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!(
                "Invalid input parameters: {}. Example: {{\"query\": \"artificial intelligence\", \"action\": \"search\"}}", e
            )))?;

        let action = params.action.as_deref().unwrap_or("search");
        let language = params.language.as_deref().unwrap_or("en");
        let limit = params.limit.unwrap_or(5).min(20).max(1);

        // Validate language code (basic validation)
        if language.len() != 2 || !language.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(Error::Other(
                "Language code must be a 2-letter lowercase code (e.g., 'en', 'es', 'fr')"
                    .to_string(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Claude-RS-Bot/1.0 (https://github.com/anthropics/claude-rs)")
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        match action {
            "search" => {
                self.search_wikipedia(&client, &params.query, language, limit)
                    .await
            }
            "summary" => {
                self.get_wikipedia_summary(&client, &params.query, language)
                    .await
            }
            _ => Err(Error::Other(
                "Invalid action. Supported actions: 'search', 'summary'".to_string(),
            )),
        }
    }
}

impl WikipediaTool {
    async fn search_wikipedia(
        &self,
        client: &reqwest::Client,
        query: &str,
        language: &str,
        limit: u32,
    ) -> Result<String> {
        let url = format!("https://{}.wikipedia.org/w/api.php", language);

        let limit_str = limit.to_string();
        let mut params = HashMap::new();
        params.insert("action", "query");
        params.insert("format", "json");
        params.insert("list", "search");
        params.insert("srsearch", query);
        params.insert("srlimit", &limit_str);
        params.insert("srprop", "snippet|wordcount");

        let response = client
            .get(&url)
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Wikipedia API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Other(format!(
                "Wikipedia API returned status: {}",
                response.status()
            )));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| Error::Other(format!("Failed to read Wikipedia response: {}", e)))?;

        let json_response: Value = serde_json::from_str(&response_text)
            .map_err(|e| Error::Other(format!("Failed to parse Wikipedia response: {}", e)))?;

        let search_results = json_response["query"]["search"]
            .as_array()
            .ok_or_else(|| Error::Other("Invalid Wikipedia search response format".to_string()))?;

        let mut results = Vec::new();
        for result in search_results {
            let title = result["title"].as_str().unwrap_or("").to_string();
            let snippet = result["snippet"]
                .as_str()
                .unwrap_or("")
                .replace("<span class=\"searchmatch\">", "")
                .replace("</span>", "");
            let wordcount = result["wordcount"].as_u64().map(|w| w as u32);

            results.push(WikipediaSearchResult {
                title,
                snippet,
                wordcount,
            });
        }

        let wiki_response = WikipediaResponse {
            action: "search".to_string(),
            query: query.to_string(),
            language: language.to_string(),
            results,
            summary: None,
        };

        serde_json::to_string_pretty(&wiki_response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }

    async fn get_wikipedia_summary(
        &self,
        client: &reqwest::Client,
        title: &str,
        language: &str,
    ) -> Result<String> {
        let url = format!("https://{}.wikipedia.org/w/api.php", language);

        let mut params = HashMap::new();
        params.insert("action", "query");
        params.insert("format", "json");
        params.insert("prop", "extracts");
        params.insert("exintro", "true");
        params.insert("explaintext", "true");
        params.insert("exsectionformat", "plain");
        params.insert("titles", title);
        params.insert("redirects", "true");

        let response = client
            .get(&url)
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Wikipedia API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Other(format!(
                "Wikipedia API returned status: {}",
                response.status()
            )));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| Error::Other(format!("Failed to read Wikipedia response: {}", e)))?;

        let json_response: Value = serde_json::from_str(&response_text)
            .map_err(|e| Error::Other(format!("Failed to parse Wikipedia response: {}", e)))?;

        let pages = json_response["query"]["pages"]
            .as_object()
            .ok_or_else(|| Error::Other("Invalid Wikipedia summary response format".to_string()))?;

        let page = pages
            .values()
            .next()
            .ok_or_else(|| Error::Other("No page found in Wikipedia response".to_string()))?;

        if page["missing"].is_boolean() {
            return Err(Error::Other(format!(
                "Wikipedia page '{}' not found",
                title
            )));
        }

        let extract = page["extract"]
            .as_str()
            .ok_or_else(|| Error::Other("No extract found in Wikipedia response".to_string()))?;

        let actual_title = page["title"].as_str().unwrap_or(title);

        // Limit summary length to prevent overly long responses
        let summary = if extract.chars().count() > 2000 {
            let truncated: String = extract.chars().take(2000).collect();
            format!("{}...", truncated)
        } else {
            extract.to_string()
        };

        let wiki_response = WikipediaResponse {
            action: "summary".to_string(),
            query: title.to_string(),
            language: language.to_string(),
            results: vec![WikipediaSearchResult {
                title: actual_title.to_string(),
                snippet: summary.clone(),
                wordcount: Some(summary.split_whitespace().count() as u32),
            }],
            summary: Some(summary),
        };

        serde_json::to_string_pretty(&wiki_response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
}
