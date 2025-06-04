use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "weather"
    }

    fn description(&self) -> &str {
        "Get current weather information for a city using Open-Meteo API"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let city = input.get("city").and_then(|v| v.as_str()).ok_or_else(|| {
            Error::Other("Missing 'city' field. Example: {\"city\": \"London\"}".to_string())
        })?;

        // First, get coordinates using geocoding API
        let geocoding_url = format!(
            "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en&format=json",
            urlencoding::encode(city)
        );

        let client = reqwest::Client::new();
        let geocoding_response = client
            .get(&geocoding_url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Failed to fetch geocoding data: {}", e)))?;

        let geocoding_data: Value = geocoding_response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse geocoding response: {}", e)))?;

        let results = geocoding_data
            .get("results")
            .and_then(|r| r.as_array())
            .ok_or_else(|| Error::Other("City not found".to_string()))?;

        if results.is_empty() {
            return Err(Error::Other("City not found".to_string()));
        }

        let location = &results[0];
        let lat = location
            .get("latitude")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Error::Other("Invalid latitude".to_string()))?;
        let lon = location
            .get("longitude")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Error::Other("Invalid longitude".to_string()))?;
        let location_name = location
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(city);
        let country = location
            .get("country")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Now get weather data
        let weather_url = format!(
            "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m&temperature_unit=celsius",
            lat, lon
        );

        let weather_response = client
            .get(&weather_url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Failed to fetch weather data: {}", e)))?;

        let weather_data: Value = weather_response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse weather response: {}", e)))?;

        let current = weather_data
            .get("current")
            .ok_or_else(|| Error::Other("No current weather data".to_string()))?;

        let temp = current
            .get("temperature_2m")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let feels_like = current
            .get("apparent_temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let humidity = current
            .get("relative_humidity_2m")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let wind_speed = current
            .get("wind_speed_10m")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let weather_code = current
            .get("weather_code")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let weather_desc = match weather_code {
            0 => "Clear sky",
            1..=3 => "Partly cloudy",
            45 | 48 => "Foggy",
            51..=57 => "Drizzle",
            61..=67 => "Rain",
            71..=77 => "Snow",
            80..=82 => "Rain showers",
            85 | 86 => "Snow showers",
            95 => "Thunderstorm",
            96 | 99 => "Thunderstorm with hail",
            _ => "Unknown",
        };

        Ok(format!(
            "Weather in {}, {}:\n\
            🌡️  Temperature: {:.1}°C (feels like {:.1}°C)\n\
            ☁️  Conditions: {}\n\
            💨 Wind: {:.1} km/h\n\
            💧 Humidity: {:.0}%",
            location_name, country, temp, feels_like, weather_desc, wind_speed, humidity
        ))
    }
}
