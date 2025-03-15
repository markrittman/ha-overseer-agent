# AI Overseer for Home Assistant

An intelligent AI agent that monitors your Home Assistant devices and provides conversational insights about your smart home.

## Features

- **AI-Powered Monitoring**: Uses Google's Gemini 2.0 FLASH LLM to monitor and analyze the state of your smart home devices
- **Voice Assistant Integration**: Ask questions about your home through Home Assistant's conversation interface
- **Real-time Insights**: View conversational commentary about what's happening in your home
- **Custom Dashboard Card**: Visual console showing the AI's thoughts and observations
- **Service API**: Programmatically query the AI about your home's status

## Installation

### HACS Installation (Recommended)

1. Make sure you have [HACS](https://hacs.xyz/) installed
2. Add this repository as a custom repository in HACS:
   - Go to HACS → Integrations → ⋮ (menu) → Custom repositories
   - Add `https://github.com/yourusername/hacs-homeassistant-overseer` with category "Integration"
3. Click "Install" on the AI Overseer integration
4. Restart Home Assistant

### Manual Installation

1. Copy the `overseer_agent` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant

## Configuration

Add the following to your `configuration.yaml`:

```yaml
overseer_agent:
  google_cloud_project_id: "your-google-cloud-project-id"
  google_cloud_credentials: "/config/google-credentials.json"
  include_domains:
    - light
    - switch
    - binary_sensor
    - climate
  exclude_domains:
    - automation
  update_interval: 30
  enable_conversation: true
  conversational_style: "thoughtful"  # Options: thoughtful, analytical, friendly, concise
```

### Configuration Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `google_cloud_project_id` | Your Google Cloud Project ID | Yes | - |
| `google_cloud_credentials` | Path to your Google Cloud credentials JSON file | Yes | - |
| `include_domains` | List of domains to monitor | No | All domains |
| `exclude_domains` | List of domains to exclude from monitoring | No | None |
| `update_interval` | Interval (in seconds) for state updates | No | 30 |
| `enable_conversation` | Enable integration with the conversation component | No | true |
| `max_insights` | Maximum number of insights to keep in history | No | 50 |
| `conversational_style` | Style of the AI's conversational output | No | thoughtful |

## Usage

### Voice Assistant

Once configured, you can ask your Home Assistant voice assistant questions about your home:

- "What's the status of my lights?"
- "Are any doors or windows open?"
- "Tell me about the temperature in my house"

### Dashboard Card

Add the AI Overseer Console card to your dashboard to see real-time insights:

1. Edit your dashboard
2. Click "Add Card" → "Custom: AI Overseer Console"
3. Configure the card (optional):
   ```yaml
   type: custom:overseer-card
   title: AI Overseer Console
   max_items: 10
   ```

### Service API

You can programmatically interact with the AI Overseer using these services:

#### overseer_agent.query

Send a direct query to the AI Overseer:

```yaml
service: overseer_agent.query
data:
  query: "What's the status of my living room lights?"
```

#### overseer_agent.analyze_entity

Request analysis of a specific entity:

```yaml
service: overseer_agent.analyze_entity
data:
  entity_id: light.living_room
```

#### overseer_agent.analyze_domain

Request analysis of all entities in a specified domain:

```yaml
service: overseer_agent.analyze_domain
data:
  domain: light
```

#### overseer_agent.clear_insights

Clear the AI Overseer's insight history:

```yaml
service: overseer_agent.clear_insights
```

## Google Cloud Setup

This integration requires a Google Cloud account with the Vertex AI API enabled:

1. Create a Google Cloud Project
2. Enable the Vertex AI API
3. Create a service account with the "Vertex AI User" role
4. Download the service account key JSON file
5. Place the JSON file in your Home Assistant config directory
6. Reference the file path in your configuration

## Troubleshooting

- Check the Home Assistant logs for errors related to the overseer_agent component
- Ensure your Google Cloud credentials are valid and have the necessary permissions
- Make sure the Vertex AI API is enabled in your Google Cloud project

## License

This project is licensed under the MIT License - see the LICENSE file for details.
