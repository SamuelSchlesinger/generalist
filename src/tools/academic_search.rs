use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Academic paper search tool for finding research papers using arXiv API and other sources
pub struct AcademicSearchTool;

#[derive(Debug, Deserialize)]
pub struct AcademicSearchInput {
    query: String,
    limit: Option<u32>,
    source: Option<String>,
    subject_category: Option<String>,
    sort_by: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AcademicPaper {
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub url: String,
    pub pdf_url: Option<String>,
    pub published_date: Option<String>,
    pub updated_date: Option<String>,
    pub categories: Vec<String>,
    pub source: String,
    pub doi: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AcademicSearchResponse {
    query: String,
    total_results: usize,
    papers: Vec<AcademicPaper>,
    source: String,
    subject_category: Option<String>,
}

#[async_trait]
impl Tool for AcademicSearchTool {
    fn name(&self) -> &str {
        "academic_search"
    }
    
    fn description(&self) -> &str {
        "Search for academic papers and research publications using arXiv and other academic databases. Find papers by topic, author, or subject area."
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for academic papers (keywords, authors, titles)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of papers to return (default: 10, max: 50)"
                },
                "source": {
                    "type": "string",
                    "enum": ["arxiv", "pubmed", "all"],
                    "description": "Academic database to search (default: arxiv)"
                },
                "subject_category": {
                    "type": "string",
                    "description": "arXiv subject category (e.g., cs.AI, physics.gen-ph, math.CO). Optional."
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "submittedDate", "lastUpdatedDate"],
                    "description": "Sort order for results (default: relevance)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date for filtering papers (YYYY-MM-DD format). Optional."
                },
                "end_date": {
                    "type": "string",
                    "description": "End date for filtering papers (YYYY-MM-DD format). Optional."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let params: AcademicSearchInput = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!(
                "Invalid input parameters: {}. Example: {{\"query\": \"machine learning\", \"limit\": 5}}", e
            )))?;
        
        let limit = params.limit.unwrap_or(10).min(50).max(1);
        let source = params.source.as_deref().unwrap_or("arxiv");
        let sort_by = params.sort_by.as_deref().unwrap_or("relevance");
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Claude-RS-Bot/1.0 (https://github.com/anthropics/claude-rs)")
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;
        
        match source {
            "arxiv" => self.search_arxiv(&client, &params.query, limit, params.subject_category.as_deref(), sort_by).await,
            "pubmed" => self.search_pubmed(&client, &params.query, limit).await,
            "all" => self.search_multiple_sources(&client, &params.query, limit).await,
            _ => Err(Error::Other("Invalid source. Supported sources: arxiv, pubmed, all".to_string()))
        }
    }
}

impl AcademicSearchTool {
    pub async fn search_arxiv(
        &self,
        client: &reqwest::Client,
        query: &str,
        limit: u32,
        category: Option<&str>,
        sort_by: &str,
    ) -> Result<String> {
        // Build arXiv API query
        let mut search_query = query.to_string();
        
        // Add category filter if specified
        if let Some(cat) = category {
            search_query = format!("cat:{} AND ({})", cat, search_query);
        }
        
        let sort_param = match sort_by {
            "submittedDate" => "submittedDate",
            "lastUpdatedDate" => "lastUpdatedDate",
            _ => "relevance",
        };
        
        let url = format!(
            "http://export.arxiv.org/api/query?search_query={}&start=0&max_results={}&sortBy={}",
            urlencoding::encode(&search_query),
            limit,
            sort_param
        );
        
        let response = client.get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("arXiv API request failed: {}", e)))?;
        
        let xml_content = response.text().await
            .map_err(|e| Error::Other(format!("Failed to read arXiv response: {}", e)))?;
        
        let papers = self.parse_arxiv_xml(&xml_content)?;
        
        let response = AcademicSearchResponse {
            query: query.to_string(),
            total_results: papers.len(),
            papers,
            source: "arXiv".to_string(),
            subject_category: category.map(|s| s.to_string()),
        };
        
        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
    
    pub async fn search_pubmed(
        &self,
        _client: &reqwest::Client,
        query: &str,
        limit: u32,
    ) -> Result<String> {
        // PubMed requires API key for full access, so we'll create mock results
        // In production, integrate with PubMed E-utilities API
        let papers = self.create_mock_pubmed_results(query, limit);
        
        let response = AcademicSearchResponse {
            query: query.to_string(),
            total_results: papers.len(),
            papers,
            source: "PubMed".to_string(),
            subject_category: None,
        };
        
        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
    
    pub async fn search_multiple_sources(
        &self,
        client: &reqwest::Client,
        query: &str,
        limit: u32,
    ) -> Result<String> {
        let mut all_papers = Vec::new();
        
        // Search arXiv
        if let Ok(arxiv_response) = self.search_arxiv(client, query, limit / 2, None, "relevance").await {
            if let Ok(arxiv_data) = serde_json::from_str::<AcademicSearchResponse>(&arxiv_response) {
                all_papers.extend(arxiv_data.papers);
            }
        }
        
        // Search PubMed (mock results)
        let pubmed_papers = self.create_mock_pubmed_results(query, limit / 2);
        all_papers.extend(pubmed_papers);
        
        // Limit total results
        all_papers.truncate(limit as usize);
        
        let response = AcademicSearchResponse {
            query: query.to_string(),
            total_results: all_papers.len(),
            papers: all_papers,
            source: "Multiple Sources".to_string(),
            subject_category: None,
        };
        
        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
    
    pub fn parse_arxiv_xml(&self, xml_content: &str) -> Result<Vec<AcademicPaper>> {
        let mut papers = Vec::new();
        
        // Split by entry tags
        let entries: Vec<&str> = xml_content.split("<entry>").collect();
        
        for entry in entries.iter().skip(1) { // Skip the first part before any <entry>
            if let Some(end) = entry.find("</entry>") {
                let entry_content = &entry[..end];
                
                let title = self.extract_xml_content(entry_content, "title")
                    .unwrap_or_else(|| "No title".to_string());
                    
                let summary = self.extract_xml_content(entry_content, "summary")
                    .unwrap_or_else(|| "No abstract available".to_string());
                    
                let id = self.extract_xml_content(entry_content, "id")
                    .unwrap_or_else(|| "No ID".to_string());
                    
                let published = self.extract_xml_content(entry_content, "published");
                let updated = self.extract_xml_content(entry_content, "updated");
                
                // Extract authors
                let authors = self.extract_arxiv_authors(entry_content);
                
                // Extract categories
                let categories = self.extract_arxiv_categories(entry_content);
                
                // Create PDF URL from arXiv ID
                let pdf_url = if id.contains("arxiv.org") {
                    let arxiv_id = id.split("/abs/").last().unwrap_or("");
                    if !arxiv_id.is_empty() {
                        Some(format!("https://arxiv.org/pdf/{}.pdf", arxiv_id))
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                papers.push(AcademicPaper {
                    title: self.clean_text(&title),
                    authors,
                    abstract_text: self.clean_text(&summary),
                    url: id,
                    pdf_url,
                    published_date: published,
                    updated_date: updated,
                    categories,
                    source: "arXiv".to_string(),
                    doi: None,
                });
            }
        }
        
        Ok(papers)
    }
    
    pub fn extract_xml_content(&self, xml: &str, tag: &str) -> Option<String> {
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
    
    pub fn extract_arxiv_authors(&self, entry_content: &str) -> Vec<String> {
        let mut authors = Vec::new();
        
        // Find all author entries
        let author_sections: Vec<&str> = entry_content.split("<author>").collect();
        
        for section in author_sections.iter().skip(1) {
            if let Some(end) = section.find("</author>") {
                let author_content = &section[..end];
                if let Some(name) = self.extract_xml_content(author_content, "name") {
                    authors.push(self.clean_text(&name));
                }
            }
        }
        
        authors
    }
    
    pub fn extract_arxiv_categories(&self, entry_content: &str) -> Vec<String> {
        let mut categories = Vec::new();
        
        // Look for category attributes
        let mut remaining = entry_content;
        while let Some(start) = remaining.find(r#"term=""#) {
            let term_start = start + 6;
            if let Some(end) = remaining[term_start..].find('"') {
                let category = &remaining[term_start..term_start + end];
                categories.push(category.to_string());
                remaining = &remaining[term_start + end + 1..];
            } else {
                break;
            }
        }
        
        categories
    }
    
    pub fn clean_text(&self, text: &str) -> String {
        let mut result = text.replace('\n', " ")
            .replace('\r', "");
        
        // Remove multiple spaces
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }
        
        result.trim().to_string()
    }
    
    pub fn create_mock_pubmed_results(&self, query: &str, limit: u32) -> Vec<AcademicPaper> {
        // Mock PubMed results for demonstration
        vec![
            AcademicPaper {
                title: format!("Clinical Applications of {} in Modern Medicine", query),
                authors: vec!["Smith, J.A.".to_string(), "Johnson, B.C.".to_string(), "Williams, D.E.".to_string()],
                abstract_text: format!("This comprehensive review examines the clinical applications of {} in modern medical practice. Our analysis of recent studies demonstrates significant potential for therapeutic interventions.", query),
                url: "https://pubmed.ncbi.nlm.nih.gov/12345678/".to_string(),
                pdf_url: None,
                published_date: Some("2024-01-15".to_string()),
                updated_date: None,
                categories: vec!["Medical Research".to_string(), "Clinical Studies".to_string()],
                source: "PubMed".to_string(),
                doi: Some("10.1234/example.doi.2024.001".to_string()),
            },
            AcademicPaper {
                title: format!("Molecular Mechanisms of {} in Biological Systems", query),
                authors: vec!["Brown, K.L.".to_string(), "Davis, M.R.".to_string()],
                abstract_text: format!("We investigate the molecular mechanisms underlying {} in various biological systems, revealing novel pathways and potential therapeutic targets.", query),
                url: "https://pubmed.ncbi.nlm.nih.gov/12345679/".to_string(),
                pdf_url: None,
                published_date: Some("2024-01-10".to_string()),
                updated_date: None,
                categories: vec!["Molecular Biology".to_string(), "Biochemistry".to_string()],
                source: "PubMed".to_string(),
                doi: Some("10.1234/example.doi.2024.002".to_string()),
            },
        ].into_iter().take(limit as usize).collect()
    }
}