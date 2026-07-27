use crate::{Error, Result, Tool};
use async_trait::async_trait;
use firecrawl::{Client, MapOptions, SitemapMode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct FirecrawlMapTool;

#[derive(Debug, Deserialize)]
pub struct FirecrawlMapInput {
    url: String,
    search: Option<String>,
    ignore_sitemap: Option<bool>,
    include_subdomains: Option<bool>,
    limit: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct FirecrawlMapResponse {
    success: bool,
    url: String,
    total_links: usize,
    links: Vec<String>,
    error: Option<String>,
}

#[async_trait]
impl Tool for FirecrawlMapTool {
    fn name(&self) -> &str {
        "firecrawl_map"
    }

    fn description(&self) -> &str {
        "Discover the URLs of a website using the Firecrawl API. Returns the list of pages \
         found via sitemaps and crawling. Useful for understanding site structure before \
         extracting specific pages."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to map"
                },
                "search": {
                    "type": "string",
                    "description": "Optional search query to filter results"
                },
                "ignore_sitemap": {
                    "type": "boolean",
                    "description": "Ignore existing sitemap.xml files"
                },
                "include_subdomains": {
                    "type": "boolean",
                    "description": "Include subdomains in the map"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of pages to map (default: 500)"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let params: FirecrawlMapInput = serde_json::from_value(input)
            .map_err(|e| Error::Tool(format!("Invalid input parameters: {}", e)))?;

        let api_key = std::env::var("FIRECRAWL_API_KEY").map_err(|_| {
            Error::Tool("FIRECRAWL_API_KEY environment variable not set".to_string())
        })?;

        let firecrawl = Client::new(&api_key)
            .map_err(|e| Error::Tool(format!("Failed to initialize Firecrawl: {:?}", e)))?;

        let mut map_options = MapOptions {
            // Unbounded maps of large sites produce enormous responses.
            limit: Some(params.limit.unwrap_or(500)),
            ..Default::default()
        };
        if let Some(search) = params.search {
            map_options.search = Some(search);
        }
        if let Some(ignore_sitemap) = params.ignore_sitemap {
            map_options.sitemap = Some(if ignore_sitemap {
                SitemapMode::Skip
            } else {
                SitemapMode::Include
            });
        }
        if let Some(include_subdomains) = params.include_subdomains {
            map_options.include_subdomains = Some(include_subdomains);
        }

        let response = match firecrawl.map_urls(&params.url, Some(map_options)).await {
            Ok(links) => FirecrawlMapResponse {
                success: true,
                url: params.url,
                total_links: links.len(),
                links,
                error: None,
            },
            Err(e) => FirecrawlMapResponse {
                success: false,
                url: params.url,
                total_links: 0,
                links: vec![],
                error: Some(format!("Map failed: {:?}", e)),
            },
        };

        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Tool(format!("Failed to serialize response: {}", e)))
    }
}
