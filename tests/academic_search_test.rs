#[cfg(test)]
mod tests {
    use claude::tools::AcademicSearchTool;
    use claude::Tool;
    use serde_json::json;
    use tokio;

    #[tokio::test]
    async fn test_academic_search_basic_functionality() {
        let tool = AcademicSearchTool;
        
        // Test basic schema
        let schema = tool.input_schema();
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("query")));
        
        // Check optional parameters
        assert!(schema["properties"]["limit"].is_object());
        assert!(schema["properties"]["source"].is_object());
        assert!(schema["properties"]["subject_category"].is_object());
    }

    #[tokio::test]
    async fn test_academic_search_tool_info() {
        let tool = AcademicSearchTool;
        
        assert_eq!(tool.name(), "academic_search");
        assert!(tool.description().contains("academic"));
        assert!(tool.description().contains("papers"));
        assert!(tool.description().contains("arXiv"));
    }

    #[tokio::test]
    async fn test_academic_search_xml_content_extraction() {
        let tool = AcademicSearchTool;
        
        let xml = r#"<entry><title>Test Paper Title</title><summary>This is the abstract</summary></entry>"#;
        
        let title = tool.extract_xml_content(xml, "title");
        assert_eq!(title, Some("Test Paper Title".to_string()));
        
        let summary = tool.extract_xml_content(xml, "summary");
        assert_eq!(summary, Some("This is the abstract".to_string()));
        
        let missing = tool.extract_xml_content(xml, "missing");
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_academic_search_author_extraction() {
        let tool = AcademicSearchTool;
        
        let entry_xml = r#"
            <author><name>John Smith</name></author>
            <author><name>Jane Doe</name></author>
            <author><name>Bob Johnson</name></author>
        "#;
        
        let authors = tool.extract_arxiv_authors(entry_xml);
        
        assert_eq!(authors.len(), 3);
        assert!(authors.contains(&"John Smith".to_string()));
        assert!(authors.contains(&"Jane Doe".to_string()));
        assert!(authors.contains(&"Bob Johnson".to_string()));
    }

    #[tokio::test]
    async fn test_academic_search_category_extraction() {
        let tool = AcademicSearchTool;
        
        let entry_xml = r#"
            <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
            <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
            <category term="stat.ML" scheme="http://arxiv.org/schemas/atom"/>
        "#;
        
        let categories = tool.extract_arxiv_categories(entry_xml);
        
        assert_eq!(categories.len(), 3);
        assert!(categories.contains(&"cs.AI".to_string()));
        assert!(categories.contains(&"cs.LG".to_string()));
        assert!(categories.contains(&"stat.ML".to_string()));
    }

    #[tokio::test]
    async fn test_academic_search_text_cleaning() {
        let tool = AcademicSearchTool;
        
        let messy_text = "This is a\n\rmessy  text\n  with multiple\r\n  spaces";
        let cleaned = tool.clean_text(messy_text);
        
        assert_eq!(cleaned, "This is a messy text with multiple spaces");
        assert!(!cleaned.contains('\n'));
        assert!(!cleaned.contains('\r'));
        assert!(!cleaned.contains("  "));
    }

    #[tokio::test]
    async fn test_academic_search_arxiv_xml_parsing() {
        let tool = AcademicSearchTool;
        
        let sample_arxiv_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2301.12345v1</id>
        <title>A Novel Approach to Machine Learning</title>
        <summary>This paper presents a novel approach to machine learning that improves upon existing methods.</summary>
        <published>2023-01-15T18:30:00Z</published>
        <updated>2023-01-16T10:15:00Z</updated>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    </entry>
    <entry>
        <id>http://arxiv.org/abs/2301.67890v1</id>
        <title>Deep Learning for Computer Vision</title>
        <summary>An exploration of deep learning techniques for computer vision applications.</summary>
        <published>2023-01-10T14:20:00Z</published>
        <author><name>Charlie Brown</name></author>
        <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
    </entry>
</feed>"#;

        let papers = tool.parse_arxiv_xml(sample_arxiv_xml).unwrap();
        
        assert_eq!(papers.len(), 2);
        
        let first_paper = &papers[0];
        assert_eq!(first_paper.title, "A Novel Approach to Machine Learning");
        assert_eq!(first_paper.authors.len(), 2);
        assert!(first_paper.authors.contains(&"Alice Smith".to_string()));
        assert!(first_paper.authors.contains(&"Bob Jones".to_string()));
        assert_eq!(first_paper.categories.len(), 2);
        assert!(first_paper.categories.contains(&"cs.AI".to_string()));
        assert_eq!(first_paper.pdf_url, Some("https://arxiv.org/pdf/2301.12345v1.pdf".to_string()));
        
        let second_paper = &papers[1];
        assert_eq!(second_paper.title, "Deep Learning for Computer Vision");
        assert_eq!(second_paper.authors.len(), 1);
        assert!(second_paper.authors.contains(&"Charlie Brown".to_string()));
    }

    #[tokio::test]
    async fn test_academic_search_pubmed_mock_results() {
        let tool = AcademicSearchTool;
        
        let papers = tool.create_mock_pubmed_results("cancer research", 3);
        
        assert!(!papers.is_empty());
        assert!(papers.len() <= 3);
        
        for paper in &papers {
            assert_eq!(paper.source, "PubMed");
            assert!(paper.title.contains("cancer research"));
            assert!(!paper.authors.is_empty());
            assert!(paper.doi.is_some());
            assert!(paper.url.contains("pubmed.ncbi.nlm.nih.gov"));
        }
    }

    #[tokio::test]
    async fn test_academic_search_input_validation() {
        let tool = AcademicSearchTool;
        
        // Test valid input
        let valid_input = json!({
            "query": "machine learning",
            "limit": 5,
            "source": "arxiv",
            "subject_category": "cs.AI"
        });
        
        // This should not panic when deserializing
        let _: Result<claude::tools::AcademicSearchInput, _> = serde_json::from_value(valid_input);
        
        // Test invalid input (missing required field)
        let invalid_input = json!({
            "limit": 5,
            "source": "arxiv"
        });
        
        let result = tool.execute(invalid_input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_academic_search_invalid_source() {
        let tool = AcademicSearchTool;
        
        let invalid_input = json!({
            "query": "test query",
            "source": "invalid_source"
        });
        
        let result = tool.execute(invalid_input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid source"));
    }

    #[tokio::test]
    async fn test_academic_search_empty_xml_parsing() {
        let tool = AcademicSearchTool;
        
        let empty_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"#;

        let papers = tool.parse_arxiv_xml(empty_xml).unwrap();
        assert_eq!(papers.len(), 0);
    }

    #[tokio::test]
    async fn test_academic_search_malformed_xml_handling() {
        let tool = AcademicSearchTool;
        
        let malformed_xml = r#"<entry><title>Incomplete entry"#;
        
        let papers = tool.parse_arxiv_xml(malformed_xml).unwrap();
        assert_eq!(papers.len(), 0);
    }

    #[tokio::test]
    async fn test_academic_search_real_arxiv_api() {
        let tool = AcademicSearchTool;
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Claude-RS-Bot/1.0 (https://github.com/anthropics/claude-rs)")
            .build()
            .unwrap();
        
        // Test real arXiv API call
        let result = tool.search_arxiv(&client, "machine learning", 3, Some("cs.AI"), "relevance").await;
        
        match result {
            Ok(response_json) => {
                let response: serde_json::Value = serde_json::from_str(&response_json).unwrap();
                
                assert_eq!(response["query"].as_str().unwrap(), "machine learning");
                assert_eq!(response["source"].as_str().unwrap(), "arXiv");
                assert_eq!(response["subject_category"].as_str().unwrap(), "cs.AI");
                
                let papers = response["papers"].as_array().unwrap();
                println!("✅ Real arXiv search returned {} papers", papers.len());
                
                // Verify paper structure
                if !papers.is_empty() {
                    let first_paper = &papers[0];
                    assert!(first_paper["title"].is_string());
                    assert!(first_paper["authors"].is_array());
                    assert!(first_paper["abstract_text"].is_string());
                    assert!(first_paper["url"].is_string());
                    assert_eq!(first_paper["source"].as_str().unwrap(), "arXiv");
                    
                    println!("First paper: {}", first_paper["title"].as_str().unwrap());
                    
                    // Check PDF URL generation
                    if let Some(pdf_url) = first_paper["pdf_url"].as_str() {
                        assert!(pdf_url.contains("arxiv.org/pdf"));
                        assert!(pdf_url.ends_with(".pdf"));
                    }
                    
                    // Check authors array
                    let authors = first_paper["authors"].as_array().unwrap();
                    if !authors.is_empty() {
                        assert!(authors[0].is_string());
                    }
                    
                    // Check categories
                    let categories = first_paper["categories"].as_array().unwrap();
                    if !categories.is_empty() {
                        assert!(categories[0].is_string());
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Real arXiv search failed: {}", e);
                // Don't fail the test as network issues are common
            }
        }
    }

    #[tokio::test]
    async fn test_academic_search_real_execution() {
        let tool = AcademicSearchTool;
        
        let input = json!({
            "query": "neural networks",
            "limit": 5,
            "source": "arxiv",
            "sort_by": "relevance"
        });
        
        let result = tool.execute(input).await;
        
        match result {
            Ok(response_json) => {
                let response: serde_json::Value = serde_json::from_str(&response_json).unwrap();
                
                assert_eq!(response["query"].as_str().unwrap(), "neural networks");
                assert_eq!(response["source"].as_str().unwrap(), "arXiv");
                
                let papers = response["papers"].as_array().unwrap();
                println!("✅ Real academic search returned {} papers", papers.len());
                
                // Should have some papers
                if !papers.is_empty() {
                    assert!(papers.len() <= 5);
                    
                    for paper in papers {
                        assert!(paper["title"].is_string());
                        assert!(paper["authors"].is_array());
                        assert!(paper["abstract_text"].is_string());
                        assert!(paper["url"].is_string());
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Real academic search failed: {}", e);
                // Don't fail the test as network issues are common
            }
        }
    }

    #[tokio::test]
    async fn test_academic_search_real_multiple_sources() {
        let tool = AcademicSearchTool;
        
        let input = json!({
            "query": "artificial intelligence",
            "limit": 6,
            "source": "all"
        });
        
        let result = tool.execute(input).await;
        
        match result {
            Ok(response_json) => {
                let response: serde_json::Value = serde_json::from_str(&response_json).unwrap();
                
                assert_eq!(response["query"].as_str().unwrap(), "artificial intelligence");
                assert_eq!(response["source"].as_str().unwrap(), "Multiple Sources");
                
                let papers = response["papers"].as_array().unwrap();
                println!("✅ Real multi-source search returned {} papers", papers.len());
                
                // Should have papers from both arXiv and mock PubMed
                if !papers.is_empty() {
                    let sources: std::collections::HashSet<&str> = papers.iter()
                        .map(|p| p["source"].as_str().unwrap())
                        .collect();
                    
                    println!("Sources found: {:?}", sources);
                    
                    // Should have at least one source
                    assert!(!sources.is_empty());
                }
            }
            Err(e) => {
                println!("⚠️  Real multi-source search failed: {}", e);
                // Don't fail the test as network issues are common
            }
        }
    }
}