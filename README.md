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
   - Add the repository URL with category "Integration"
3. Click "Install" on the AI Overseer integration
4. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/overseer_agent` folder to your Home Assistant `custom_components` directory
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

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Required APIs**:
   - In the Cloud Console, go to "APIs & Services" → "Library"
   - Enable the following APIs:
     - **Vertex AI API** (`aiplatform.googleapis.com`)
     - **Cloud Storage API** (`storage.googleapis.com`) - may be required as a dependency

3. **Create Service Account with Required Permissions**:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Enter a name (e.g., "ha-overseer-agent")
   - Click "Create and Continue"
   - Assign the following roles:
     - **Vertex AI User** (`roles/aiplatform.user`) - Required for making inference requests
     - **Vertex AI Service Agent** (`roles/aiplatform.serviceAgent`) - Required for service account to act on behalf of Vertex AI

4. **Create and Download Key**:
   - Find your service account in the list
   - Click the three dots (⋮) menu → "Manage keys"
   - Click "Add Key" → "Create new key"
   - Select "JSON" format and download the key file

5. **Add Key to Home Assistant**:
   - Upload the JSON key file to your Home Assistant configuration directory
   - Reference it in your configuration as shown above

### Billing Considerations

- Vertex AI requires a billing account to be linked to your Google Cloud project
- The Gemini 2.0 FLASH model has associated costs based on usage
- Consider setting up budget alerts to monitor your spending

### Troubleshooting Permission Issues

If you encounter permission errors:

- Check the Home Assistant logs for specific error messages
- Verify that the service account has the correct roles assigned
- Ensure the key file is accessible to Home Assistant
- Confirm that the APIs are enabled in your Google Cloud project
- Make sure your Google Cloud project has billing enabled

## Troubleshooting

- Check the Home Assistant logs for errors related to the overseer_agent component
- Ensure your Google Cloud credentials are valid and have the necessary permissions
- Make sure the Vertex AI API is enabled in your Google Cloud project

### Common Issues

#### "Integration was not setup via the UI" Message

If you see a message saying "This integration was not setup via the UI, you have either set it up in YAML or it is a dependency set up by another integration," this is normal behavior. Since the Overseer Agent is configured in your `configuration.yaml` file, it cannot be configured through the UI. This message is informational, not an error.

To verify the integration is working correctly:
1. Check if the Overseer Agent services are available in the Services tab of Developer Tools
2. Try using the conversation component to ask about your home
3. Look for insights being generated in the Lovelace card (if installed)

#### Google Cloud Credentials Issues

If you see errors related to Google Cloud credentials:
1. Ensure the credentials file exists at the path specified in your configuration
2. Verify the project ID matches your Google Cloud project
3. Check that the service account has the necessary permissions for Vertex AI
4. Make sure the Vertex AI API is enabled and billing is set up for your project

## Compatibility

This integration is compatible with Home Assistant 2023.3.0 and newer. It has been tested with the following versions:

- Home Assistant 2023.3.0
- Home Assistant 2023.6.0
- Home Assistant 2023.9.0
- Home Assistant 2023.12.0
- Home Assistant 2024.3.0

### LangChain Compatibility

This integration supports both older and newer versions of LangChain:
- For LangChain < 0.1.0: Uses the original module structure
- For LangChain >= 0.1.0: Uses the new module structure with `langchain_community` and `langchain_google_vertexai`

### Home Assistant Compatibility

This integration supports different versions of Home Assistant:
- Works with both older and newer conversation API implementations
- Handles different versions of the Lovelace resource registration API

### Pydantic Compatibility

This integration is compatible with Pydantic v2, which is used by newer versions of LangChain.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
