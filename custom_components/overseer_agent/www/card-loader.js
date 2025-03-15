const CARD_NAME = "overseer-card";

class OverseerCardLoader {
  static async loadCard() {
    if (customElements.get(CARD_NAME)) return;
    
    console.info(`%c OVERSEER-CARD %c Loading custom card...`, 
      "color: white; background: #4CAF50; font-weight: 700;",
      "color: #4CAF50; background: white; font-weight: 700;");

    // Import the lit-element modules
    await Promise.all([
      customElements.whenDefined("home-assistant-main"),
      customElements.whenDefined("hui-view")
    ]);

    // Load the card
    await import("./overseer-card.js");
    
    // Register with custom cards
    window.customCards = window.customCards || [];
    window.customCards.push({
      type: CARD_NAME,
      name: "AI Overseer Console",
      description: "A card that displays the AI Overseer's insights in real-time"
    });

    console.info(`%c OVERSEER-CARD %c Custom card loaded successfully!`, 
      "color: white; background: #4CAF50; font-weight: 700;",
      "color: #4CAF50; background: white; font-weight: 700;");
  }
}

// Load the card
OverseerCardLoader.loadCard().catch(error => {
  console.error(`%c OVERSEER-CARD %c Error loading card: ${error}`, 
    "color: white; background: #f44336; font-weight: 700;",
    "color: #f44336; background: white; font-weight: 700;");
});
