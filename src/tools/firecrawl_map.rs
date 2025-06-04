use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use firecrawl::FirecrawlApp;
use firecrawl::map::MapOptions;
use std::collections::HashMap;

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
    sitemap: Vec<SitemapEntry>,
    link_graph: HashMap<String, Vec<String>>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SitemapEntry {
    url: String,
    title: Option<String>,
    description: Option<String>,
    last_modified: Option<String>,
    content_type: Option<String>,
    size: Option<usize>,
}

#[async_trait]
impl Tool for FirecrawlMapTool {
    fn name(&self) -> &str {
        "firecrawl_map"
    }
    
    fn description(&self) -> &str {
        "Map website structure using Firecrawl API - discovers all pages and links within a website, creating a comprehensive sitemap. Useful for understanding site architecture and finding all available pages."
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
                    "description": "Maximum number of pages to map"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let params: FirecrawlMapInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!("Invalid input parameters: {}", e)))?;
        
        let api_key = std::env::var("FIRECRAWL_API_KEY")
            .map_err(|_| Error::Other("FIRECRAWL_API_KEY environment variable not set".to_string()))?;
        
        let firecrawl = FirecrawlApp::new(&api_key)
            .map_err(|e| Error::Other(format!("Failed to initialize Firecrawl: {:?}", e)))?;
        
        let mut map_options = MapOptions::default();
        
        if let Some(search) = params.search {
            map_options.search = Some(search);
        }
        
        if let Some(ignore_sitemap) = params.ignore_sitemap {
            map_options.ignore_sitemap = Some(ignore_sitemap);
        }
        
        if let Some(include_subdomains) = params.include_subdomains {
            map_options.include_subdomains = Some(include_subdomains);
        }
        
        if let Some(limit) = params.limit {
            map_options.limit = Some(limit);
        }
        
        match firecrawl.map_url(&params.url, Some(map_options)).await {
            Ok(map_result) => {
                let mut link_graph: HashMap<String, Vec<String>> = HashMap::new();
                let mut sitemap: Vec<SitemapEntry> = Vec::new();
                
                for link in &map_result {
                    let entry = SitemapEntry {
                        url: link.clone(),
                        title: None,
                        description: None,
                        last_modified: None,
                        content_type: None,
                        size: None,
                    };
                    sitemap.push(entry);
                    
                    if !link_graph.contains_key(link) {
                        link_graph.insert(link.clone(), Vec::new());
                    }
                }
                
                if map_result.len() > 1 {
                    for (i, source) in map_result.iter().enumerate() {
                        for (j, target) in map_result.iter().enumerate() {
                            if i != j && source.contains(&params.url) && target.contains(&params.url) {
                                link_graph.get_mut(source).unwrap().push(target.clone());
                            }
                        }
                    }
                }
                
                let response = FirecrawlMapResponse {
                    success: true,
                    url: params.url,
                    total_links: sitemap.len(),
                    sitemap,
                    link_graph,
                    error: None,
                };
                
                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
            }
            Err(e) => {
                let response = FirecrawlMapResponse {
                    success: false,
                    url: params.url,
                    total_links: 0,
                    sitemap: vec![],
                    link_graph: HashMap::new(),
                    error: Some(format!("Map failed: {:?}", e)),
                };
                
                serde_json::to_string_pretty(&response)
                    .map_err(|e| Error::Other(format!("Failed to serialize error response: {}", e)))
            }
        }
    }
}