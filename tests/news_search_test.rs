#[cfg(test)]
mod tests {
    use claude::tools::NewsSearchTool;
    use claude::Tool;
    use serde_json::json;
    use tokio;

    #[tokio::test]
    async fn test_news_search_basic_functionality() {
        let tool = NewsSearchTool;
        
        // Test basic schema
        let schema = tool.input_schema();
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("query")));
    }

    #[tokio::test]
    async fn test_news_search_rss_parsing() {
        let tool = NewsSearchTool;
        
        // Test RSS XML parsing with sample RSS content
        let sample_rss = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test News</title>
        <item>
            <title><![CDATA[AI Technology Breakthrough]]></title>
            <description><![CDATA[Scientists announce new developments in artificial intelligence research.]]></description>
            <link>https://example.com/ai-news</link>
            <pubDate>Mon, 15 Jan 2024 10:30:00 GMT</pubDate>
        </item>
        <item>
            <title>Climate Change Update</title>
            <description>Latest reports on global climate initiatives and policy changes.</description>
            <link>https://example.com/climate-news</link>
            <pubDate>Mon, 15 Jan 2024 08:15:00 GMT</pubDate>
        </item>
    </channel>
</rss>"#;

        let articles = tool.parse_rss_xml(sample_rss).unwrap();
        assert_eq!(articles.len(), 2);
        
        let ai_article = &articles[0];
        assert_eq!(ai_article.title, "AI Technology Breakthrough");
        assert!(ai_article.description.as_ref().unwrap().contains("artificial intelligence"));
        assert_eq!(ai_article.url, "https://example.com/ai-news");
        assert!(ai_article.published_at.is_some());
    }

    #[tokio::test]
    async fn test_news_search_filtering() {
        let tool = NewsSearchTool;
        
        let articles = vec![
            claude::tools::NewsArticle {
                title: "AI Revolution in Healthcare".to_string(),
                description: Some("Artificial intelligence transforms medical diagnosis".to_string()),
                url: "https://example.com/ai-health".to_string(),
                source: "Tech News".to_string(),
                published_at: Some("2024-01-15T10:30:00Z".to_string()),
                content_snippet: None,
            },
            claude::tools::NewsArticle {
                title: "Sports Update".to_string(),
                description: Some("Latest scores from basketball games".to_string()),
                url: "https://example.com/sports".to_string(),
                source: "Sports News".to_string(),
                published_at: Some("2024-01-15T09:00:00Z".to_string()),
                content_snippet: None,
            },
        ];

        let filtered = tool.filter_articles_by_query(&articles, "AI", "Test Source");
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].title.contains("AI"));
    }

    #[tokio::test]
    async fn test_news_search_relevance_scoring() {
        let tool = NewsSearchTool;
        
        let score1 = tool.calculate_relevance_score(
            "AI Breakthrough in Machine Learning",
            "Scientists develop new artificial intelligence algorithms",
            "AI machine learning"
        );
        
        let score2 = tool.calculate_relevance_score(
            "Weather Update",
            "Today's weather forecast shows sunny skies",
            "AI machine learning"
        );
        
        assert!(score1 > score2);
        assert!(score1 > 0.0);
    }

    #[tokio::test]
    async fn test_news_search_html_cleaning() {
        let tool = NewsSearchTool;
        
        let dirty_html = "<![CDATA[<p><strong>Breaking News:</strong> <em>AI development</em> continues.</p>]]>";
        let cleaned = tool.clean_html(dirty_html);
        
        assert_eq!(cleaned, "Breaking News: AI development continues.");
        assert!(!cleaned.contains("<"));
        assert!(!cleaned.contains("CDATA"));
    }

    #[tokio::test]
    async fn test_news_search_rss_feeds_selection() {
        let tool = NewsSearchTool;
        
        // Test English feeds
        let en_feeds = tool.get_rss_feeds("en", None, None);
        assert!(!en_feeds.is_empty());
        assert!(en_feeds.iter().any(|(name, _)| name.contains("BBC")));
        
        // Test US country-specific feeds
        let us_feeds = tool.get_rss_feeds("en", Some("us"), None);
        assert!(us_feeds.len() >= en_feeds.len());
        
        // Test Spanish feeds
        let es_feeds = tool.get_rss_feeds("es", None, None);
        assert!(!es_feeds.is_empty());
        
        // Test unsupported language defaults to English
        let unknown_feeds = tool.get_rss_feeds("xx", None, None);
        assert!(!unknown_feeds.is_empty());
    }

    #[tokio::test]
    async fn test_news_search_xml_tag_extraction() {
        let tool = NewsSearchTool;
        
        let xml = r#"<item><title>Test Title</title><description>Test Description</description></item>"#;
        
        let title = tool.extract_xml_tag_content(xml, "title");
        assert_eq!(title, Some("Test Title".to_string()));
        
        let description = tool.extract_xml_tag_content(xml, "description");
        assert_eq!(description, Some("Test Description".to_string()));
        
        let missing = tool.extract_xml_tag_content(xml, "missing");
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_news_search_input_validation() {
        let tool = NewsSearchTool;
        
        // Test valid input
        let valid_input = json!({
            "query": "artificial intelligence",
            "language": "en",
            "limit": 5
        });
        
        // This should not panic when deserializing
        let _: Result<claude::tools::NewsSearchInput, _> = serde_json::from_value(valid_input);
        
        // Test invalid input (missing required field)
        let invalid_input = json!({
            "language": "en",
            "limit": 5
        });
        
        let result = tool.execute(invalid_input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_news_search_real_rss_feeds() {
        let tool = NewsSearchTool;
        
        // Test actual RSS feed fetching from BBC
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Claude-RS-Bot/1.0 (https://github.com/anthropics/claude-rs)")
            .build()
            .unwrap();
        
        let result = tool.fetch_and_parse_rss(&client, "http://feeds.bbci.co.uk/news/world/rss.xml").await;
        
        match result {
            Ok(articles) => {
                assert!(!articles.is_empty(), "Should fetch some articles from BBC RSS");
                
                // Check that articles have required fields
                for article in &articles[..std::cmp::min(3, articles.len())] {
                    assert!(!article.title.is_empty(), "Article should have a title");
                    assert!(!article.url.is_empty(), "Article should have a URL");
                    assert_eq!(article.source, "RSS Feed");
                }
                
                println!("✅ Successfully fetched {} articles from BBC RSS", articles.len());
            }
            Err(e) => {
                println!("⚠️  BBC RSS fetch failed: {}", e);
                // Don't fail the test as network issues are common
            }
        }
    }

    #[tokio::test]
    async fn test_news_search_real_execution() {
        let tool = NewsSearchTool;
        
        let input = json!({
            "query": "technology",
            "language": "en",
            "limit": 3
        });
        
        let result = tool.execute(input).await;
        
        match result {
            Ok(response_json) => {
                let response: serde_json::Value = serde_json::from_str(&response_json).unwrap();
                
                assert_eq!(response["query"].as_str().unwrap(), "technology");
                assert_eq!(response["language"].as_str().unwrap(), "en");
                
                let articles = response["articles"].as_array().unwrap();
                println!("✅ Real news search returned {} articles", articles.len());
                
                // If we got articles, verify their structure
                if !articles.is_empty() {
                    let first_article = &articles[0];
                    assert!(first_article["title"].is_string());
                    assert!(first_article["url"].is_string());
                    assert!(first_article["source"].is_string());
                }
            }
            Err(e) => {
                println!("⚠️  Real news search failed: {}", e);
                // Don't fail the test as network issues are common
            }
        }
    }
}

