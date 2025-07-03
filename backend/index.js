const express = require('express');
const mainRouter = require('./routes/mainRouter'); 
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3001;

app.use('/static', express.static(path.join(__dirname, 'public')));
app.use(cors());
app.use(express.json());

app.use('/', mainRouter); 

app.listen(PORT, () => {
    console.log(`Frame Capture Server running at http://localhost:${PORT}`);
    // console.log('Serving static files from:', path.join(__dirname, 'public'));

});
