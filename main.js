const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');
const fs = require('fs').promises;
const { OpenAIEmbeddings } = require('@langchain/openai');
const { FaissStore } = require('@langchain/community/vectorstores/faiss');
const dotenv = require('dotenv');

dotenv.config();

// Load the OpenAI API key from the environment
const { OPENAI_API_KEY } = process.env;

// Set the API key for Langchain
process.env.OPENAI_API_KEY = OPENAI_API_KEY;

const app = express();
const port = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY
});

// Global variables
let cvDataset;
let vectorStore;
let isInitialized = false;

// Load CV dataset and initialize vector store
async function initialize() {
  try {
    const data = await fs.readFile('cv_dataset.json', 'utf8');
    cvDataset = JSON.parse(data);
    console.log('CV dataset loaded successfully');

    const embeddings = new OpenAIEmbeddings();
    vectorStore = await FaissStore.fromTexts(
      cvDataset.map(item => `${item.question} ${item.answer}`),
      cvDataset.map((_, i) => ({ id: i })),
      embeddings
    );
    console.log('Vector store initialized successfully');
    isInitialized = true;
  } catch (error) {
    console.error('Error during initialization:', error);
    throw error;
  }
}

// Find relevant examples
async function findRelevantExamples(question, numExamples = 3) {
  if (!isInitialized || !vectorStore) {
    console.error('Vector store is not initialized');
    return [];
  }
  try {
    const results = await vectorStore.similaritySearch(question, numExamples);
    return results.map(result => cvDataset[result.metadata.id]);
  } catch (error) {
    console.error('Error finding relevant examples:', error);
    return [];
  }
}

// Prepare prompt
async function preparePrompt(question) {
  let prompt = "Relevant information from Mathieu Vialatte's CV:\n\n";
  
  if (isInitialized) {
    const relevantExamples = await findRelevantExamples(question);
    relevantExamples.forEach(example => {
      prompt += `Q: ${example.question}\nA: ${example.answer}\n\n`;
    });
  } else {
    prompt += "No relevant examples could be found. Please answer to the best of your ability with general information about Mathieu Vialatte.\n\n";
  }
  
  prompt += `Question: ${question}\nAnswer:`;
  return prompt;
}

// Main chat route
app.post('/chat', async (req, res) => {
  try {
    const { message, conversationHistory = [] } = req.body;
    console.log('Received message:', message);

    const prompt = await preparePrompt(message);
    
    const fullConversation = [
      { role: "system", content: `You are a virtual assistant specialized in answering questions about Mathieu Vialatte's CV. 
        Use only the information provided in the examples or in the question to answer. 
        If information is not available, clearly state so. 
        Focus on Mathieu's skills, experience, and achievements. 
        Respond concisely and professionally.` },
      ...conversationHistory,
      { role: "user", content: prompt }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: fullConversation,
      max_tokens: 250,
      temperature: 0.5,
      top_p: 0.9,
      presence_penalty: 0.6,
      frequency_penalty: 0.2,
    });

    const aiResponse = response.choices[0].message.content.trim();
    console.log('AI response:', aiResponse);

    // Update conversation history
    conversationHistory.push({ role: "user", content: message });
    conversationHistory.push({ role: "assistant", content: aiResponse });

    res.json({ 
      response: aiResponse,
      conversationHistory 
    });
  } catch (error) {
    console.error('An error occurred:', error);
    res.status(500).json({ response: "Sorry, an error occurred while processing your request." });
  }
});

// Health check route
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    initialized: isInitialized
  });
});

// Initialize and start the server
initialize()
  .then(() => {
    app.listen(port, () => {
      console.log(`Server listening on port ${port}`);
    });
  })
  .catch(error => {
    console.error('Fatal error during initialization:', error);
    process.exit(1);
  });