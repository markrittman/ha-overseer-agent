import { LitElement, html, css } from "https://unpkg.com/lit-element@2.4.0/lit-element.js?module";

class OverseerCard extends LitElement {
  static get properties() {
    return {
      hass: { type: Object },
      config: { type: Object },
      insights: { type: Array },
      _connected: { type: Boolean },
      _unsubscribe: { type: Function },
    };
  }

  constructor() {
    super();
    this.insights = [];
    this._connected = false;
    this._unsubscribe = null;
  }

  static get styles() {
    return css`
      :host {
        display: block;
        padding: 16px;
      }
      .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
      }
      .card-header h2 {
        margin: 0;
        font-size: 18px;
        font-weight: 500;
      }
      .card-header .actions {
        display: flex;
      }
      .card-header .actions button {
        background: none;
        border: none;
        padding: 8px;
        cursor: pointer;
        color: var(--primary-text-color);
        border-radius: 50%;
      }
      .card-header .actions button:hover {
        background: var(--secondary-background-color);
      }
      .console {
        background-color: var(--card-background-color, #000);
        border-radius: 4px;
        padding: 16px;
        height: 300px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        color: var(--primary-text-color, #0f0);
        border: 1px solid var(--divider-color);
      }
      .insight {
        margin-bottom: 16px;
        border-bottom: 1px solid var(--divider-color);
        padding-bottom: 8px;
      }
      .insight:last-child {
        border-bottom: none;
      }
      .insight-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-bottom: 4px;
      }
      .insight-content {
        white-space: pre-wrap;
      }
      .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--secondary-text-color);
      }
      .empty-state ha-icon {
        margin-bottom: 8px;
        --mdi-icon-size: 40px;
      }
      .connection-error {
        color: var(--error-color);
        margin-top: 8px;
        text-align: center;
      }
    `;
  }

  setConfig(config) {
    if (!config) {
      throw new Error("Invalid configuration");
    }
    this.config = {
      title: config.title || "AI Overseer Console",
      max_items: config.max_items || 10,
    };
  }

  getCardSize() {
    return 4;
  }

  connectedCallback() {
    super.connectedCallback();
    this._connect();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._disconnect();
  }

  _connect() {
    if (this._connected || !this.hass) return;

    const wsConnection = this.hass.connection;
    
    wsConnection.subscribeMessage(
      (msg) => this._handleMessage(msg),
      { type: "overseer_agent/subscribe" }
    ).then((unsub) => {
      this._unsubscribe = unsub;
      this._connected = true;
      this._fetchInsights();
    }).catch((err) => {
      console.error("Error subscribing to overseer insights:", err);
      this._connected = false;
    });
  }

  _disconnect() {
    if (this._unsubscribe) {
      this._unsubscribe();
      this._unsubscribe = null;
    }
    this._connected = false;
  }

  _fetchInsights() {
    if (!this.hass || !this._connected) return;

    this.hass.connection.sendMessagePromise({
      type: "overseer_agent/insights",
      count: this.config.max_items
    }).then((result) => {
      if (result && result.insights) {
        this.insights = result.insights;
      }
    }).catch((err) => {
      console.error("Error fetching overseer insights:", err);
    });
  }

  _handleMessage(msg) {
    if (msg && msg.insights) {
      this.insights = msg.insights.slice(-this.config.max_items);
    }
  }

  _refreshInsights() {
    this._fetchInsights();
  }

  _clearInsights() {
    if (!this.hass) return;
    
    this.hass.callService("overseer_agent", "clear_insights", {});
  }

  _formatTimestamp(timestamp) {
    if (!timestamp) return "";
    
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  _getSourceIcon(source) {
    switch (source) {
      case "system":
        return "mdi:cog";
      case "event_analysis":
        return "mdi:magnify";
      case "query":
        return "mdi:help-circle";
      case "conversation":
        return "mdi:message-text";
      case "entity_analysis":
        return "mdi:chart-line";
      case "domain_analysis":
        return "mdi:view-grid";
      default:
        return "mdi:robot";
    }
  }

  _getSourceLabel(source) {
    switch (source) {
      case "system":
        return "System";
      case "event_analysis":
        return "Event Analysis";
      case "query":
        return "Query";
      case "conversation":
        return "Conversation";
      case "entity_analysis":
        return "Entity Analysis";
      case "domain_analysis":
        return "Domain Analysis";
      default:
        return source;
    }
  }

  render() {
    if (!this.hass) {
      return html`
        <ha-card>
          <div class="card-header">
            <h2>${this.config.title}</h2>
          </div>
          <div class="console">
            <div class="empty-state">
              <ha-icon icon="mdi:robot-off"></ha-icon>
              <div>Home Assistant connection not available</div>
            </div>
          </div>
        </ha-card>
      `;
    }

    return html`
      <ha-card>
        <div class="card-header">
          <h2>${this.config.title}</h2>
          <div class="actions">
            <button @click=${this._refreshInsights} title="Refresh">
              <ha-icon icon="mdi:refresh"></ha-icon>
            </button>
            <button @click=${this._clearInsights} title="Clear">
              <ha-icon icon="mdi:delete"></ha-icon>
            </button>
          </div>
        </div>
        <div class="console">
          ${this.insights.length === 0 ? html`
            <div class="empty-state">
              <ha-icon icon="mdi:robot"></ha-icon>
              <div>No insights available yet</div>
              ${!this._connected ? html`
                <div class="connection-error">
                  Not connected to the Overseer Agent
                </div>
              ` : ''}
            </div>
          ` : html`
            ${this.insights.slice().reverse().map((insight) => html`
              <div class="insight">
                <div class="insight-header">
                  <div>
                    <ha-icon icon="${this._getSourceIcon(insight.source)}"></ha-icon>
                    ${this._getSourceLabel(insight.source)}
                  </div>
                  <div>${this._formatTimestamp(insight.timestamp)}</div>
                </div>
                <div class="insight-content">${insight.content}</div>
              </div>
            `)}
          `}
        </div>
      </ha-card>
    `;
  }
}

// Don't define the custom element here, let the card-loader.js handle it
// This prevents duplicate registration errors
if (!customElements.get('overseer-card')) {
  customElements.define("overseer-card", OverseerCard);
}

window.customCards = window.customCards || [];
window.customCards.push({
  type: "overseer-card",
  name: "AI Overseer Console",
  description: "A card that displays the AI Overseer's insights in real-time"
});
