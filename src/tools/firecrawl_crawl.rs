use crate::{Error, Result, Tool};
use async_trait::async_trait;
use firecrawl::crawl::{CrawlOptions, CrawlScrapeOptions};
use firecrawl::FirecrawlApp;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct FirecrawlCrawlTool;

#[derive(Debug, Deserialize)]
pub struct FirecrawlCrawlInput {
    url: String,
    max_depth: Option<u32>,
    limit: Option<u32>,
    exclude_patterns: Option<Vec<String>>,
    include_patterns: Option<Vec<String>>,
    allow_backward_links: Option<bool>,
    allow_external_links: Option<bool>,
    headers: Option<std::collections::HashMap<String, String>>,
    wait_for: Option<u32>,
    timeout: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct FirecrawlCrawlResponse {
    success: bool,
    total_pages: usize,
    completed_pages: usize,
    pages: Vec<CrawledPage>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CrawledPage {
    url: String,
    title: Option<String>,
    content: Option<String>,
    markdown: Option<String>,
    html: Option<String>,
    links: Option<Vec<String>>,
    metadata: Option<Value>,
}

#[async_trait]
impl Tool for FirecrawlCrawlTool {
    fn name(&self) -> &str {
        "firecrawl_crawl"
    }

    fn description(&self) -> &str {
        "Crawl websites using Firecrawl API - a powerful web scraping service that handles JavaScript rendering, anti-bot measures, and content extraction. Can crawl entire websites or specific sections based on URL patterns."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to start crawling from"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to crawl (default: 2)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of pages to crawl"
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL patterns to exclude from crawling"
                },
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL patterns to include in crawling"
                },
                "allow_backward_links": {
                    "type": "boolean",
                    "description": "Allow crawling pages that link back to parent pages"
                },
                "allow_external_links": {
                    "type": "boolean",
                    "description": "Allow crawling external links"
                },
                "headers": {
                    "type": "object",
                    "description": "Custom headers to send with requests"
                },
                "wait_for": {
                    "type": "integer",
                    "description": "Time to wait for page to load (milliseconds)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout (milliseconds)"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let params: FirecrawlCrawlInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!("Invalid input parameters: {}", e)))?;

        let api_key = std::env::var("FIRECRAWL_API_KEY").map_err(|_| {
            Error::Other("FIRECRAWL_API_KEY environment variable not set".to_string())
        })?;

        let firecrawl = FirecrawlApp::new(&api_key)
            .map_err(|e| Error::Other(format!("Failed to initialize Firecrawl: {:?}", e)))?;

        let mut scrape_options = CrawlScrapeOptions::default();

        if let Some(headers) = params.headers {
            scrape_options.headers = Some(headers);
        }

        if let Some(wait_for) = params.wait_for {
            scrape_options.wait_for = Some(wait_for);
        }

        if let Some(timeout) = params.timeout {
            scrape_options.timeout = Some(timeout);
        }

        let mut crawl_options = CrawlOptions::default();
        crawl_options.scrape_options = Some(scrape_options);

        if let Some(max_depth) = params.max_depth {
            crawl_options.max_depth = Some(max_depth);
        }

        if let Some(limit) = params.limit {
            crawl_options.limit = Some(limit);
        }

        if let Some(exclude) = params.exclude_patterns {
            crawl_options.exclude_paths = Some(exclude);
        }

        if let Some(include) = params.include_patterns {
            crawl_options.include_paths = Some(include);
        }

        if let Some(allow_backward) = params.allow_backward_links {
            crawl_options.allow_backward_links = Some(allow_backward);
        }

        if let Some(allow_external) = params.allow_external_links {
            crawl_options.allow_external_links = Some(allow_external);
        }

        match firecrawl.crawl_url(&params.url, Some(crawl_options)).await {
            Ok(crawl_result) => {
                let pages: Vec<CrawledPage> = crawl_result
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, doc)| CrawledPage {
                        url: format!("page_{}", i), // Documents don't have URLs in crawl results
                        title: doc.metadata.title.clone(),
                        content: doc.markdown.clone(),
                        markdown: doc.markdown,
                        html: doc.html,
                        links: doc.links,
                        metadata: Some(serde_json::to_value(&doc.metadata).unwrap_or(Value::Null)),
                    })
                    .collect();

                let response = FirecrawlCrawlResponse {
                    success: true,
                    total_pages: crawl_result.total as usize,
                    completed_pages: crawl_result.completed as usize,
                    pages,
                    error: None,
                };

                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
            }
            Err(e) => {
                let response = FirecrawlCrawlResponse {
                    success: false,
                    total_pages: 0,
                    completed_pages: 0,
                    pages: vec![],
                    error: Some(format!("Crawl failed: {:?}", e)),
                };

                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize error response: {}", e)))
            }
        }
    }
}
