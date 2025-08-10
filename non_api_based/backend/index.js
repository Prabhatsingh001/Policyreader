// index.js
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const cors = require('cors');
const redis = require('./redisClient');
const { v4: uuidv4 } = require('uuid');

const app = express();
const port = 3000;
const SESSION_TTL_SECONDS = 300;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Setup multer storage for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = 'uploads/';
    if (!fs.existsSync(uploadPath)) {
      fs.mkdirSync(uploadPath);
    }
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});
const upload = multer({ storage });

// ---------------- Session Management ----------------

app.get('/api/chat/initiate', async (req, res) => {
  const session_id = `session_${uuidv4()}`;
  const sessionData = {
    created_at: Date.now(),
    lastActive: Date.now(),
    messages: [],
  };

  try {
    await redis.set(session_id, JSON.stringify(sessionData), { EX: SESSION_TTL_SECONDS });
    res.status(200).json({
      message: 'Chat session successfully initiated.',
      sessionId: session_id,
    });
  } catch (error) {
    console.error('Redis error:', error);
    res.status(500).json({ error: 'Failed to create session' });
  }
});

app.get('/api/chat/validate/:sessionId', async (req, res) => {
  const sessionId = req.params.sessionId;
  if (!sessionId) return res.status(400).json({ error: "No sessionId provided" });

  try {
    const sessionData = await redis.get(sessionId);
    if (!sessionData) return res.status(404).json({ valid: false });

    await redis.expire(sessionId, SESSION_TTL_SECONDS); // Refresh TTL
    res.status(200).json({ valid: true });
  } catch (error) {
    console.error(`Failed to validate session:`, error);
    res.status(500).json({ error: "Failed to validate session" });
  }
});

// ---------------- File Upload Endpoint ----------------

app.post('/upload', upload.single('file'), async (req, res) => {
  const file = req.file;
  const sessionId = req.body.sessionId;

  if (!file) return res.status(400).json({ error: 'No file uploaded' });
  if (!sessionId) return res.status(400).json({ error: 'No session ID provided' });

  try {
    const sessionData = await redis.get(sessionId);
    if (sessionData) {
      const parsed = JSON.parse(sessionData);
      parsed.uploads = parsed.uploads || [];
      parsed.uploads.push({ filename: file.filename, originalName: file.originalname });
      parsed.lastActive = Date.now(); // Refresh session activity
      await redis.set(sessionId, JSON.stringify(parsed));
    }

    res.status(200).json({
      message: 'File uploaded successfully',
      filename: file.filename,
    });
  } catch (err) {
    console.error('Upload error:', err);
    res.status(500).json({ error: 'Upload failed' });
  }
});

// ---------------- Chat Message Endpoint ----------------

app.post('/chat/send', upload.array('documents', 5), async (req, res) => {
  const { chat } = req.body;
  const files = req.files || [];
  const sessionId = req.body.sessionId;
  if (!sessionId) return res.status(400).json({ error: "Missing sessionId" });

  try {
    // Refresh TTL and update last active
    const sessionData = await redis.get(sessionId);
    if (sessionData) {
      const parsed = JSON.parse(sessionData);
      parsed.lastActive = Date.now();
      parsed.messages = parsed.messages || [];
      parsed.messages.push({ sender: 'user', text: chat, files: files.map(f => f.originalname) });
      await redis.set(sessionId, JSON.stringify(parsed), { EX: SESSION_TTL_SECONDS });
    }

    const formData = new FormData();
    formData.append('query', chat);
    formData.append('top_k', 5);
    formData.append('threshold', 0.3);

    files.forEach((file) => {
      formData.append('files', fs.createReadStream(file.path), file.originalname);
    });

    const aiServiceUrl = 'http://localhost:8000/query';
    const aiResponse = await axios.post(aiServiceUrl, formData, {
      headers: formData.getHeaders()
    });

    // Cleanup temp uploads
    files.forEach(file => fs.unlinkSync(file.path));

    res.status(200).json({
      ai_response: aiResponse.data.answer || "No answer received.",
      files: aiResponse.data.files || []
    });

  } catch (error) {
    console.error('Error in /chat/send:', error.message);
    if (error.response) console.error('Response error:', error.response.data);
    res.status(500).json({ ai_response: 'Error communicating with AI backend.' });
  }
});

// ---------------- Mock Proxy Routes ----------------

app.post('/load-documents', async (req, res) => {
  try {
    const resp = await axios.post("http://localhost:8000/load-documents");
    res.json(resp.data);
  } catch (err) {
    res.status(500).json({ error: 'Failed to load documents' });
  }
});

app.post('/batch-query', async (req, res) => {
  try {
    const resp = await axios.post("http://localhost:8000/batch-query", req.body);
    res.json({
      responses: resp.data.responses.map(r => ({ ai_response: r.answer, files: r.files || [] })),
      total_queries: resp.data.total_queries
    });
  } catch (err) {
    res.status(500).json({ error: 'Batch query failed' });
  }
});

// ---------------- Start Server ----------------

app.listen(port, () => {
  console.log(`Express chat server listening at http://localhost:${port}`);
});