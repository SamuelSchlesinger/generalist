use crate::{Error, Result, Tool};
use async_trait::async_trait;
use firecrawl::{Client, SearchOptions, SearchResultOrDocument};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct FirecrawlSearchTool;

#[derive(Debug, Deserialize)]
pub struct FirecrawlSearchInput {
    query: String,
    limit: Option<u32>,
    location: Option<String>,
    tbs: Option<String>,
    include_domains: Option<Vec<String>>,
    exclude_domains: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct FirecrawlSearchResponse {
    success: bool,
    query: String,
    total_results: usize,
    results: Vec<SearchResult>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    title: String,
    url: String,
    description: String,
}

#[async_trait]
impl Tool for FirecrawlSearchTool {
    fn name(&self) -> &str {
        "firecrawl_search"
    }

    fn description(&self) -> &str {
        "Search the web using Firecrawl API - a powerful web scraping service that searches and extracts clean, structured content from web pages. Unlike basic search, this returns the actual page content."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)"
                },
                "location": {
                    "type": "string",
                    "description": "Location for local search results (e.g., 'Germany', 'San Francisco, CA')"
                },
                "tbs": {
                    "type": "string",
                    "description": "Time-based search parameter (e.g., 'qdr:d' for past day)"
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only return results from these domains"
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let params: FirecrawlSearchInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!("Invalid input parameters: {}", e)))?;

        let api_key = std::env::var("FIRECRAWL_API_KEY").map_err(|_| {
            Error::Other("FIRECRAWL_API_KEY environment variable not set".to_string())
        })?;

        let firecrawl = Client::new(&api_key)
            .map_err(|e| Error::Other(format!("Failed to initialize Firecrawl: {:?}", e)))?;

        let search_options = SearchOptions {
            limit: params.limit,
            location: params.location,
            tbs: params.tbs,
            include_domains: params.include_domains,
            exclude_domains: params.exclude_domains,
            timeout: Some(60000),
            ..Default::default()
        };

        match firecrawl.search(&params.query, Some(search_options)).await {
            Ok(search_result) => {
                let results: Vec<SearchResult> = search_result
                    .data
                    .web
                    .unwrap_or_default()
                    .into_iter()
                    .map(|entry| match entry {
                        SearchResultOrDocument::WebResult(web) => SearchResult {
                            title: web.title.unwrap_or_default(),
                            url: web.url,
                            description: web.description.unwrap_or_default(),
                        },
                        // The API returns full documents when scraping is enabled;
                        // fall back to their metadata for the summary fields.
                        SearchResultOrDocument::Document(doc) => {
                            let meta = doc.metadata.unwrap_or_default();
                            SearchResult {
                                title: meta.title.unwrap_or_default(),
                                url: meta.source_url.unwrap_or_default(),
                                description: meta.description.unwrap_or_default(),
                            }
                        }
                    })
                    .collect();

                let response = FirecrawlSearchResponse {
                    success: true,
                    query: params.query,
                    total_results: results.len(),
                    results,
                    error: None,
                };

                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
            }
            Err(e) => {
                let response = FirecrawlSearchResponse {
                    success: false,
                    query: params.query,
                    total_results: 0,
                    results: vec![],
                    error: Some(format!("Search failed: {:?}", e)),
                };

                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize error response: {}", e)))
            }
        }
    }
}
