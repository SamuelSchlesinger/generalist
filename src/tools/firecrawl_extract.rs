use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use firecrawl::FirecrawlApp;
use firecrawl::scrape::{ScrapeOptions, ScrapeFormats, JsonOptions};

/// Firecrawl Extract Tool - Extracts structured content from web pages
/// 
/// This tool supports AI-powered structured data extraction using JSON schemas.
/// When an `extract_schema` is provided, Firecrawl will use an LLM to extract
/// data matching the schema from the page content.
/// 
/// Example extract_schema:
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "company_name": {"type": "string"},
///     "pricing": {
///       "type": "array",
///       "items": {
///         "type": "object",
///         "properties": {
///           "plan": {"type": "string"},
///           "price": {"type": "string"}
///         }
///       }
///     }
///   }
/// }
/// ```
pub struct FirecrawlExtractTool;

#[derive(Debug, Deserialize)]
pub struct FirecrawlExtractInput {
    url: String,
    formats: Option<Vec<String>>,
    only_main_content: Option<bool>,
    include_tags: Option<Vec<String>>,
    exclude_tags: Option<Vec<String>>,
    headers: Option<std::collections::HashMap<String, String>>,
    wait_for: Option<u32>,
    timeout: Option<u32>,
    extract_schema: Option<Value>,
}


#[derive(Debug, Serialize)]
pub struct FirecrawlExtractResponse {
    success: bool,
    url: String,
    title: Option<String>,
    content: Option<String>,
    markdown: Option<String>,
    html: Option<String>,
    extracted_data: Option<Value>,
    links: Option<Vec<String>>,
    images: Option<Vec<String>>,
    metadata: Option<PageMetadata>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PageMetadata {
    title: Option<String>,
    description: Option<String>,
    language: Option<String>,
    keywords: Option<String>,
    author: Option<String>,
    published_date: Option<String>,
    modified_date: Option<String>,
    site_name: Option<String>,
    og_data: Option<Value>,
}

#[async_trait]
impl Tool for FirecrawlExtractTool {
    fn name(&self) -> &str {
        "firecrawl_extract"
    }
    
    fn description(&self) -> &str {
        "Extract clean, structured content from web pages using Firecrawl API - handles JavaScript rendering, removes ads/popups, and can extract data according to custom schemas. Supports multiple output formats including AI-powered structured data extraction using JSON schemas."
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract data from"
                },
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["markdown", "html", "rawHtml", "content", "links", "screenshot", "screenshot@fullPage"]},
                    "description": "Formats to extract (default: ['markdown', 'content', 'links'])"
                },
                "only_main_content": {
                    "type": "boolean",
                    "description": "Extract only the main content area"
                },
                "include_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "HTML tags to include in extraction"
                },
                "exclude_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "HTML tags to exclude from extraction"
                },
                "headers": {
                    "type": "object",
                    "description": "Custom headers to send with the request"
                },
                "wait_for": {
                    "type": "integer",
                    "description": "Time to wait for page to load (milliseconds)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout (milliseconds)"
                },
                "extract_schema": {
                    "type": "object",
                    "description": "JSON schema for AI-powered structured data extraction. When provided, Firecrawl will use LLM to extract data matching this schema from the page content. The extracted data will be available in the 'extracted_data' field of the response. Example schema: {\"type\": \"object\", \"properties\": {\"company_name\": {\"type\": \"string\"}, \"pricing\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"plan\": {\"type\": \"string\"}, \"price\": {\"type\": \"string\"}}}}}}"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let params: FirecrawlExtractInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!("Invalid input parameters: {}", e)))?;
        
        let api_key = std::env::var("FIRECRAWL_API_KEY")
            .map_err(|_| Error::Other("FIRECRAWL_API_KEY environment variable not set".to_string()))?;
        
        let firecrawl = FirecrawlApp::new(&api_key)
            .map_err(|e| Error::Other(format!("Failed to initialize Firecrawl: {:?}", e)))?;
        
        let mut scrape_options = ScrapeOptions::default();
        
        if let Some(formats) = params.formats {
            let mut scrape_formats = Vec::new();
            for format in formats {
                match format.as_str() {
                    "markdown" => scrape_formats.push(ScrapeFormats::Markdown),
                    "html" => scrape_formats.push(ScrapeFormats::HTML),
                    "rawHtml" => scrape_formats.push(ScrapeFormats::RawHTML),
                    "links" => scrape_formats.push(ScrapeFormats::Links),
                    "screenshot" => scrape_formats.push(ScrapeFormats::Screenshot),
                    "screenshot@fullPage" => scrape_formats.push(ScrapeFormats::ScreenshotFullPage),
                    _ => {}
                }
            }
            if !scrape_formats.is_empty() {
                scrape_options.formats = Some(scrape_formats);
            }
        }
        
        if let Some(only_main) = params.only_main_content {
            scrape_options.only_main_content = Some(only_main);
        }
        
        if let Some(include) = params.include_tags {
            scrape_options.include_tags = Some(include);
        }
        
        if let Some(exclude) = params.exclude_tags {
            scrape_options.exclude_tags = Some(exclude);
        }
        
        if let Some(headers) = params.headers {
            scrape_options.headers = Some(headers);
        }
        
        if let Some(wait_for) = params.wait_for {
            scrape_options.wait_for = Some(wait_for);
        }
        
        if let Some(timeout) = params.timeout {
            scrape_options.timeout = Some(timeout);
        }
        
        // Handle extract_schema for structured data extraction
        if let Some(schema) = params.extract_schema {
            // Add Json format to enable LLM extraction
            let mut formats = scrape_options.formats.unwrap_or_default();
            if !formats.iter().any(|f| matches!(f, ScrapeFormats::Json)) {
                formats.push(ScrapeFormats::Json);
            }
            scrape_options.formats = Some(formats);
            
            // Set up JSON extraction options
            let json_options = JsonOptions {
                schema: Some(schema),
                system_prompt: None,
                prompt: None,
                agent: None,
            };
            scrape_options.json_options = Some(json_options);
        }
        
        match firecrawl.scrape_url(&params.url, Some(scrape_options)).await {
            Ok(scrape_result) => {
                let metadata = Some(&scrape_result.metadata);
                let metadata = if let Some(meta) = metadata {
                    Some(PageMetadata {
                        title: meta.title.clone(),
                        description: meta.description.clone(),
                        language: meta.language.clone(),
                        keywords: meta.keywords.clone(),
                        author: None, // DocumentMetadata doesn't have author field
                        published_date: meta.published_time.clone(),
                        modified_date: meta.modified_time.clone(),
                        site_name: meta.og_site_name.clone(),
                        og_data: None,
                    })
                } else {
                    None
                };
                
                let images = None;
                
                let response = FirecrawlExtractResponse {
                    success: true,
                    url: params.url,
                    title: scrape_result.metadata.title.clone(),
                    content: scrape_result.markdown.clone(),
                    markdown: scrape_result.markdown,
                    html: scrape_result.html,
                    extracted_data: scrape_result.extract,
                    links: scrape_result.links,
                    images,
                    metadata,
                    error: None,
                };
                
                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
            }
            Err(e) => {
                let response = FirecrawlExtractResponse {
                    success: false,
                    url: params.url,
                    title: None,
                    content: None,
                    markdown: None,
                    html: None,
                    extracted_data: None,
                    links: None,
                    images: None,
                    metadata: None,
                    error: Some(format!("Extract failed: {:?}", e)),
                };
                
                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize error response: {}", e)))
            }
        }
    }
}