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
        color: var(--primary-text-color);
      }
      .console {
        background-color: var(--card-background-color, #1c1c1c);
        border-radius: 8px;
        padding: 16px;
        height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        color: var(--primary-text-color);
        display: flex;
        flex-direction: column-reverse;
      }
      .insight {
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--divider-color, rgba(255, 255, 255, 0.1));
        animation: fadeIn 0.5s ease-in;
      }
      .insight-content {
        margin-top: 4px;
        white-space: pre-wrap;
        line-height: 1.5;
      }
      .insight-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-bottom: 4px;
      }
      .insight-source {
        text-transform: uppercase;
        font-weight: bold;
      }
      .insight-timestamp {
        opacity: 0.7;
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
        --mdc-icon-size: 40px;
      }
      .controls {
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
      }
      .controls button {
        background: var(--primary-color);
        color: var(--text-primary-color);
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 14px;
        cursor: pointer;
        margin-left: 8px;
      }
      .controls button.secondary {
        background: var(--secondary-background-color);
        color: var(--primary-text-color);
      }
      .source-system {
        color: var(--info-color, #4285f4);
      }
      .source-event_analysis {
        color: var(--success-color, #0f9d58);
      }
      .source-query, .source-conversation {
        color: var(--warning-color, #f4b400);
      }
      .source-entity_analysis, .source-domain_analysis {
        color: var(--accent-color, #db4437);
      }
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .status-indicator {
        display: flex;
        align-items: center;
        font-size: 14px;
      }
      .status-indicator ha-icon {
        margin-right: 4px;
        --mdc-icon-size: 16px;
      }
      .connected {
        color: var(--success-color, #0f9d58);
      }
      .disconnected {
        color: var(--error-color, #db4437);
      }
    `;
  }

  setConfig(config) {
    if (!config) {
      throw new Error("Invalid configuration");
    }
    this.config = {
      title: "AI Overseer Console",
      max_items: 10,
      ...config,
    };
  }

  getCardSize() {
    return 5;
  }

  connectedCallback() {
    super.connectedCallback();
    this._connected = false;
    this._fetchInsights();
    this._subscribeToInsights();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribeFromInsights();
  }

  async _fetchInsights() {
    if (!this.hass) return;

    try {
      const connection = this.hass.connection;
      const response = await connection.sendMessagePromise({
        type: "overseer_agent/insights",
        count: this.config.max_items || 10,
      });

      if (response && response.insights) {
        this.insights = response.insights;
      }
    } catch (error) {
      console.error("Error fetching insights:", error);
    }
  }

  _subscribeToInsights() {
    if (!this.hass || this._connected) return;

    const connection = this.hass.connection;
    connection.subscribeMessage(
      (message) => {
        if (message.insights) {
          this.insights = message.insights;
        }
      },
      { type: "overseer_agent/subscribe" }
    ).then((unsub) => {
      this._unsubscribe = unsub;
      this._connected = true;
      this.requestUpdate();
    }).catch((err) => {
      console.error("Error subscribing to insights:", err);
      this._connected = false;
      this.requestUpdate();
    });
  }

  _unsubscribeFromInsights() {
    if (this._unsubscribe) {
      this._unsubscribe();
      this._unsubscribe = null;
    }
    this._connected = false;
  }

  _clearInsights() {
    if (!this.hass) return;
    
    this.hass.callService("overseer_agent", "clear_insights", {});
  }

  _formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  _getSourceClass(source) {
    return `source-${source.replace(/\s+/g, '_').toLowerCase()}`;
  }

  _getSourceIcon(source) {
    const icons = {
      system: "mdi:cog",
      event_analysis: "mdi:magnify",
      query: "mdi:message-question",
      conversation: "mdi:message-text",
      entity_analysis: "mdi:eye",
      domain_analysis: "mdi:database"
    };
    
    return icons[source] || "mdi:information";
  }

  _getSourceLabel(source) {
    const labels = {
      system: "System",
      event_analysis: "Analysis",
      query: "Query",
      conversation: "Conversation",
      entity_analysis: "Entity Analysis",
      domain_analysis: "Domain Analysis"
    };
    
    return labels[source] || source;
  }

  render() {
    if (!this.hass || !this.config) {
      return html``;
    }

    const sortedInsights = [...this.insights].sort((a, b) => {
      return new Date(b.timestamp) - new Date(a.timestamp);
    }).slice(0, this.config.max_items || 10);

    return html`
      <ha-card>
        <div class="card-header">
          <h2>${this.config.title}</h2>
          <div class="status-indicator ${this._connected ? 'connected' : 'disconnected'}">
            <ha-icon icon="${this._connected ? 'mdi:wifi' : 'mdi:wifi-off'}"></ha-icon>
            ${this._connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
        
        <div class="console">
          ${sortedInsights.length === 0 
            ? html`
                <div class="empty-state">
                  <ha-icon icon="mdi:robot-outline"></ha-icon>
                  <div>No insights available yet. The AI Overseer will share its thoughts here as it monitors your home.</div>
                </div>
              ` 
            : sortedInsights.map(insight => html`
                <div class="insight">
                  <div class="insight-header">
                    <div class="insight-source ${this._getSourceClass(insight.source)}">
                      <ha-icon icon="${this._getSourceIcon(insight.source)}"></ha-icon>
                      ${this._getSourceLabel(insight.source)}
                    </div>
                    <div class="insight-timestamp">${this._formatTimestamp(insight.timestamp)}</div>
                  </div>
                  <div class="insight-content">${insight.content}</div>
                </div>
              `)
          }
        </div>
        
        <div class="controls">
          <button class="secondary" @click="${this._fetchInsights}">
            <ha-icon icon="mdi:refresh"></ha-icon> Refresh
          </button>
          <button @click="${this._clearInsights}">
            <ha-icon icon="mdi:delete"></ha-icon> Clear
          </button>
        </div>
      </ha-card>
    `;
  }
}

customElements.define("overseer-card", OverseerCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: "overseer-card",
  name: "AI Overseer Console",
  description: "A card that displays the AI Overseer's insights in real-time"
});
