use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// News search tool for finding recent news articles using RSS feeds and web scraping
pub struct NewsSearchTool;

#[derive(Debug, Deserialize)]
pub struct NewsSearchInput {
    query: String,
    language: Option<String>,
    country: Option<String>,
    limit: Option<u32>,
    sources: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NewsArticle {
    pub title: String,
    pub description: Option<String>,
    pub url: String,
    pub source: String,
    pub published_at: Option<String>,
    pub content_snippet: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct NewsSearchResponse {
    query: String,
    total_results: usize,
    articles: Vec<NewsArticle>,
    language: String,
    country: Option<String>,
    sources_searched: Vec<String>,
}

#[async_trait]
impl Tool for NewsSearchTool {
    fn name(&self) -> &str {
        "news_search"
    }
    
    fn description(&self) -> &str {
        "Search for recent news articles from RSS feeds and news sources. Find breaking news, headlines, and articles by topic."
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for news articles"
                },
                "language": {
                    "type": "string",
                    "description": "Language code (default: en). Examples: en, es, fr, de, it, pt, ru, zh"
                },
                "country": {
                    "type": "string", 
                    "description": "Country code for regional news (e.g., us, gb, ca, au, de, fr). Optional."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of articles to return (default: 5, max: 20)"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of news sources to search (e.g., [\"bbc\", \"reuters\", \"cnn\"])"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let params: NewsSearchInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!(
                "Invalid input parameters: {}. Example: {{\"query\": \"artificial intelligence\", \"language\": \"en\"}}", e
            )))?;
        
        let language = params.language.as_deref().unwrap_or("en");
        let limit = params.limit.unwrap_or(5).min(20).max(1);
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Claude-RS-Bot/1.0 (https://github.com/anthropics/claude-rs)")
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;
        
        // Search using RSS feeds and web scraping
        self.search_news_rss(&client, &params.query, language, params.country.as_deref(), limit, params.sources).await
    }
}

impl NewsSearchTool {
    pub async fn search_news_rss(
        &self,
        client: &reqwest::Client,
        query: &str,
        language: &str,
        country: Option<&str>,
        limit: u32,
        sources: Option<Vec<String>>,
    ) -> Result<String> {
        let rss_feeds = self.get_rss_feeds(language, country, sources);
        let mut all_articles = Vec::new();
        let mut sources_searched = Vec::new();
        
        // Search through RSS feeds
        for (source_name, feed_url) in rss_feeds.iter().take(5) { // Limit to 5 sources to avoid timeout
            sources_searched.push(source_name.clone());
            
            match self.fetch_and_parse_rss(client, feed_url).await {
                Ok(articles) => {
                    let filtered_articles = self.filter_articles_by_query(&articles, query, source_name);
                    all_articles.extend(filtered_articles);
                }
                Err(e) => {
                    eprintln!("Failed to fetch RSS from {}: {}", source_name, e);
                    continue;
                }
            }
        }
        
        // Sort by relevance (basic keyword matching) and limit results
        all_articles.sort_by(|a, b| {
            let a_score = self.calculate_relevance_score(&a.title, &a.description.as_deref().unwrap_or(""), query);
            let b_score = self.calculate_relevance_score(&b.title, &b.description.as_deref().unwrap_or(""), query);
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        all_articles.truncate(limit as usize);
        
        let response = NewsSearchResponse {
            query: query.to_string(),
            total_results: all_articles.len(),
            articles: all_articles,
            language: language.to_string(),
            country: country.map(|s| s.to_string()),
            sources_searched,
        };
        
        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
    
    pub fn get_rss_feeds(&self, language: &str, country: Option<&str>, _sources: Option<Vec<String>>) -> Vec<(String, String)> {
        let mut feeds = Vec::new();
        
        // Default RSS feeds based on language and country
        match language {
            "en" => {
                feeds.extend(vec![
                    ("BBC World".to_string(), "http://feeds.bbci.co.uk/news/world/rss.xml".to_string()),
                    ("Reuters World".to_string(), "https://feeds.reuters.com/reuters/worldNews".to_string()),
                    ("Reuters Tech".to_string(), "https://feeds.reuters.com/reuters/technologyNews".to_string()),
                    ("AP News".to_string(), "https://feeds.apnews.com/rss/apf-topnews".to_string()),
                    ("NPR News".to_string(), "https://feeds.npr.org/1001/rss.xml".to_string()),
                ]);
                
                if country == Some("us") {
                    feeds.extend(vec![
                        ("CNN Top Stories".to_string(), "http://rss.cnn.com/rss/edition.rss".to_string()),
                        ("NBC News".to_string(), "https://feeds.nbcnews.com/nbcnews/public/news".to_string()),
                    ]);
                }
            }
            "es" => {
                feeds.extend(vec![
                    ("El País".to_string(), "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada".to_string()),
                    ("BBC Mundo".to_string(), "https://feeds.bbci.co.uk/mundo/rss.xml".to_string()),
                ]);
            }
            "fr" => {
                feeds.extend(vec![
                    ("Le Monde".to_string(), "https://www.lemonde.fr/rss/une.xml".to_string()),
                    ("BBC Afrique".to_string(), "https://feeds.bbci.co.uk/afrique/rss.xml".to_string()),
                ]);
            }
            "de" => {
                feeds.extend(vec![
                    ("Deutsche Welle".to_string(), "https://rss.dw.com/xml/rss-en-all".to_string()),
                    ("BBC Germany".to_string(), "https://feeds.bbci.co.uk/news/world/europe/rss.xml".to_string()),
                ]);
            }
            _ => {
                // Default to English feeds for unsupported languages
                feeds.extend(vec![
                    ("BBC World".to_string(), "http://feeds.bbci.co.uk/news/world/rss.xml".to_string()),
                    ("Reuters World".to_string(), "https://feeds.reuters.com/reuters/worldNews".to_string()),
                ]);
            }
        }
        
        feeds
    }
    
    pub async fn fetch_and_parse_rss(&self, client: &reqwest::Client, feed_url: &str) -> Result<Vec<NewsArticle>> {
        let response = client.get(feed_url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Failed to fetch RSS feed: {}", e)))?;
        
        let rss_text = response.text().await
            .map_err(|e| Error::Other(format!("Failed to read RSS content: {}", e)))?;
        
        self.parse_rss_xml(&rss_text)
    }
    
    pub fn parse_rss_xml(&self, xml_content: &str) -> Result<Vec<NewsArticle>> {
        // Simple XML parsing for RSS - in production, use a proper XML parser like `quick-xml`
        let mut articles = Vec::new();
        
        // Extract items using basic string matching (this is simplified)
        let items: Vec<&str> = xml_content.split("<item>").collect();
        
        for item in items.iter().skip(1) { // Skip the first part before any <item>
            if let Some(end) = item.find("</item>") {
                let item_content = &item[..end];
                
                let title = self.extract_xml_tag_content(item_content, "title")
                    .unwrap_or_else(|| "No title".to_string());
                let description = self.extract_xml_tag_content(item_content, "description");
                let link = self.extract_xml_tag_content(item_content, "link")
                    .unwrap_or_else(|| "No link".to_string());
                let pub_date = self.extract_xml_tag_content(item_content, "pubDate");
                
                articles.push(NewsArticle {
                    title: self.clean_html(&title),
                    description: description.map(|d| self.clean_html(&d)),
                    url: link,
                    source: "RSS Feed".to_string(),
                    published_at: pub_date,
                    content_snippet: None,
                });
            }
        }
        
        Ok(articles)
    }
    
    pub fn extract_xml_tag_content(&self, xml: &str, tag: &str) -> Option<String> {
        let start_tag = format!("<{}>", tag);
        let end_tag = format!("</{}>", tag);
        
        if let Some(start) = xml.find(&start_tag) {
            let content_start = start + start_tag.len();
            if let Some(end) = xml[content_start..].find(&end_tag) {
                return Some(xml[content_start..content_start + end].to_string());
            }
        }
        None
    }
    
    pub fn clean_html(&self, text: &str) -> String {
        // Basic HTML tag removal
        let mut result = text.to_string();
        
        // Remove CDATA sections
        if result.contains("<![CDATA[") {
            result = result.replace("<![CDATA[", "").replace("]]>", "");
        }
        
        // Remove common HTML tags
        let tags_to_remove = ["<p>", "</p>", "<br>", "<br/>", "<strong>", "</strong>", "<em>", "</em>", "<b>", "</b>", "<i>", "</i>"];
        for tag in &tags_to_remove {
            result = result.replace(tag, "");
        }
        
        // Clean up multiple spaces and newlines
        result = result.replace("\n", " ").replace("\r", "");
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }
        
        result.trim().to_string()
    }
    
    pub fn filter_articles_by_query(&self, articles: &[NewsArticle], query: &str, source: &str) -> Vec<NewsArticle> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        
        articles.iter()
            .filter(|article| {
                let title_lower = article.title.to_lowercase();
                let desc_lower = article.description.as_deref().unwrap_or("").to_lowercase();
                let combined_text = format!("{} {}", title_lower, desc_lower);
                
                // Check if any query term appears in the article
                query_terms.iter().any(|term| combined_text.contains(term))
            })
            .map(|article| NewsArticle {
                title: article.title.clone(),
                description: article.description.clone(),
                url: article.url.clone(),
                source: source.to_string(),
                published_at: article.published_at.clone(),
                content_snippet: article.content_snippet.clone(),
            })
            .collect()
    }
    
    pub fn calculate_relevance_score(&self, title: &str, description: &str, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let title_lower = title.to_lowercase();
        let desc_lower = description.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        
        let mut score = 0.0;
        
        for term in &query_terms {
            // Title matches are more important
            if title_lower.contains(term) {
                score += 2.0;
            }
            // Description matches
            if desc_lower.contains(term) {
                score += 1.0;
            }
        }
        
        score
    }
}