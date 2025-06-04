use crate::{Error, Result, Tool};
use async_trait::async_trait;
use firecrawl::search::SearchParams;
use firecrawl::FirecrawlApp;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct FirecrawlSearchTool;

#[derive(Debug, Deserialize)]
pub struct FirecrawlSearchInput {
    query: String,
    limit: Option<u32>,
    lang: Option<String>,
    country: Option<String>,
    location: Option<String>,
    tbs: Option<String>,
    filter: Option<String>,
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
                "lang": {
                    "type": "string",
                    "description": "Language code (e.g., 'en', 'es', 'fr')"
                },
                "country": {
                    "type": "string",
                    "description": "Country code (e.g., 'us', 'uk', 'ca')"
                },
                "location": {
                    "type": "string",
                    "description": "Location for local search results"
                },
                "tbs": {
                    "type": "string",
                    "description": "Time-based search parameter (e.g., 'qdr:d' for past day)"
                },
                "filter": {
                    "type": "string",
                    "description": "Additional search filters"
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

        let firecrawl = FirecrawlApp::new(&api_key)
            .map_err(|e| Error::Other(format!("Failed to initialize Firecrawl: {:?}", e)))?;

        let search_params = SearchParams {
            query: params.query.clone(),
            limit: params.limit,
            lang: params.lang.or(Some("en".to_string())),
            country: params.country.or(Some("us".to_string())),
            location: params.location,
            tbs: params.tbs,
            filter: params.filter,
            origin: Some("api".to_string()),
            timeout: Some(60000),
            scrape_options: None,
        };

        match firecrawl.search_with_params(search_params).await {
            Ok(search_result) => {
                let results: Vec<SearchResult> = search_result
                    .data
                    .into_iter()
                    .map(|doc| SearchResult {
                        title: doc.title,
                        url: doc.url,
                        description: doc.description,
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
