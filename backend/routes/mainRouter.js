const express = require('express');
const fs = require('fs');
const path = require('path');
const http = require('http');

const mainRouter = new express.Router();

const MJPEG_URL = 'http://localhost:3000/video';
const FRAMES_DIR = path.join(__dirname, 'frames');
const UNVERIFIED_DIR = path.join(__dirname, '..', 'public', 'unverified');

if (!fs.existsSync(FRAMES_DIR)) {
  fs.mkdirSync(FRAMES_DIR);
}

// Helper to generate safe filename from timestamp
function getTimeStampFilename() {
  return new Date().toISOString().replace(/:/g, '-');
}

// Function to capture frames and save prompt
function captureFramesFromStream(url, frameLimit = 5, prompt = '') {
  return new Promise((resolve, reject) => {
    let buffer = Buffer.alloc(0);
    let frameCount = 0;
    let savedFrames = [];

    const request = http.get(url, (res) => {
      res.on('data', (chunk) => {
        buffer = Buffer.concat([buffer, chunk]);

        const start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
        const end = buffer.indexOf(Buffer.from([0xff, 0xd9]));

        if (start !== -1 && end !== -1 && end > start) {
          const jpgBuffer = buffer.slice(start, end + 2);
          buffer = buffer.slice(end + 2);

          const timestamp = getTimeStampFilename();
          const imageFilename = `${timestamp}.jpg`;
          const imagePath = path.join(FRAMES_DIR, imageFilename);
          fs.writeFileSync(imagePath, jpgBuffer);

          // Write prompt to corresponding .txt file
          const promptFilename = `${timestamp}.txt`;
          const promptPath = path.join(FRAMES_DIR, promptFilename);
          fs.writeFileSync(promptPath, prompt);

          savedFrames.push({
            image: imageFilename,
            promptFile: promptFilename
          });

          console.log(`Saved ${imageFilename} and ${promptFilename}`);
          frameCount++;

          if (frameCount >= frameLimit) {
            res.destroy();
            resolve(savedFrames);
          }
        }
      });

      res.on('end', () => {
        resolve(savedFrames);
      });

      res.on('error', (err) => {
        reject(err);
      });
    });

    request.on('error', (err) => {
      reject(err);
    });
  });
}

// Endpoint to capture frames and save prompt
mainRouter.get('/capture_frames', async (req, res) => {
  const numFrames = parseInt(req.query.count) || 20;
  const prompt = req.query.prompt || '';

  if (!prompt.trim()) {
    return res.status(400).json({
      status: 'error',
      message: 'Missing required prompt in query string.'
    });
  }

  try {
    const frames = await captureFramesFromStream(MJPEG_URL, numFrames, prompt);
    res.json({
      status: 'success',
      message: `Captured ${frames.length} frame(s)`,
      files: frames
    });
  } catch (err) {
    console.error('Error capturing frames:', err);
    res.status(500).json({
      status: 'error',
      message: 'Failed to capture frames',
      error: err.toString()
    });
  }
});

// //Route to get the unverified images

mainRouter.get('/unverified_images', async (req, res) => {
  const UNVERIFIED_DIR = path.join(__dirname, '..', 'public', 'unverified');
  if (!fs.existsSync(UNVERIFIED_DIR)) {
    return res.status(404).json({
      status: 'error',
      message: 'Unverified folder not found.'
    });
  }

  try {
    const files = fs.readdirSync(UNVERIFIED_DIR);
    const imageFiles = files.filter(f => f.endsWith('.jpg'));

    const results = [];

    for (const imageFile of imageFiles) {
      const baseName = path.parse(imageFile).name;
      const imgPath = path.join(UNVERIFIED_DIR, imageFile);
      const promptPath = path.join(UNVERIFIED_DIR, `${baseName}_prompt.txt`);
      const bboxPath = path.join(UNVERIFIED_DIR, `${baseName}_bbox.txt`);

      // ✅ Check: only proceed if all three files exist
      if (
        fs.existsSync(imgPath) &&
        fs.existsSync(promptPath) &&
        fs.existsSync(bboxPath)
      ) {
        const prompt = fs.readFileSync(promptPath, 'utf-8').trim();
        const bboxText = fs.readFileSync(bboxPath, 'utf-8').trim();
        const [x, y, width, height] = bboxText.split(/[\s,]+/).map(Number); // robust splitter

        results.push({
          imagePath: `unverified/${imageFile}`,
          boundingBox: { x, y, width, height },
          prompt
        });
      } else {
        console.warn(`Skipping ${imageFile} due to missing prompt/bbox/image.`);
      }
    }

    return res.json(results);

  } catch (err) {
    console.error('Error reading unverified images:', err);
    return res.status(500).json({
      status: 'error',
      message: 'Failed to load unverified images',
      error: err.toString()
    });
  }
});


//API route to delete images.

mainRouter.post('/deleteImage', (req, res) => {
  const { imagePath } = req.body;

  if (!Array.isArray(imagePath) || imagePath.length === 0) {
    return res.status(400).json({
      success: false,
      message: "'imagePath' must be a non-empty array",
    });
  }

  const deleted = [];
  const notFound = [];
  const errors = [];

  // ✅ Base path: public/unverified
  const UNVERIFIED_DIR = path.join(__dirname, '..', 'public', 'unverified');

  imagePath.forEach((relativePath) => {
    try {
      const fullImagePath = path.join(UNVERIFIED_DIR, path.basename(relativePath));
      const baseName = path.parse(fullImagePath).name;

      const promptPath = path.join(UNVERIFIED_DIR, `${baseName}_prompt.txt`);
      const bboxPath = path.join(UNVERIFIED_DIR, `${baseName}_bbox.txt`);

      // Image
      if (fs.existsSync(fullImagePath)) {
        fs.unlinkSync(fullImagePath);
        deleted.push(relativePath);
      } else {
        notFound.push(relativePath);
      }

      // Prompt
      if (fs.existsSync(promptPath)) {
        fs.unlinkSync(promptPath);
      } else {
        console.warn(`Prompt not found: ${promptPath}`);
      }

      // BBox
      if (fs.existsSync(bboxPath)) {
        fs.unlinkSync(bboxPath);
      } else {
        console.warn(`BBox not found: ${bboxPath}`);
      }

    } catch (err) {
      console.error(`Error deleting files for ${relativePath}:`, err);
      errors.push({ path: relativePath, error: err.toString() });
    }
  });

  const success = errors.length === 0;

  return res.status(success ? 200 : 207).json({
    success,
    message: success ? "All files deleted successfully." : "Some deletions failed.",
    deleted,
    notFound,
    errors,
  });
});



//Final API route to send the verified images from the frontend to the backend 

const VERIFIED_DIR = path.join(__dirname, 'verified');

// Ensure required folders exist
[FRAMES_DIR, VERIFIED_DIR, UNVERIFIED_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  // Static file server for viewing frames and prompt files
  // mainRouter.post('/verified', (req, res) => {
  //     const verifiedImages = req.body;

  //     if (!Array.isArray(verifiedImages) || verifiedImages.length === 0) {
  //         return res.status(400).json({
  //             success: false,
  //             message: "Send an array of verified image data."
  //         });
  //     }

  //     try {
  //         verifiedImages.forEach(item => {
  //             const { imagePath, boundingBox, prompt } = item;
  //             console.log(imagePath,boundingBox,prompt);
  //             if (!imagePath || !boundingBox || !prompt) {
  //                 console.warn("Skipping item due to missing data:", item);
  //                 return;
  //             }

  //             const srcPath = path.join(__dirname, imagePath); // e.g. unverified/image.jpg
  //             const imageName = path.basename(srcPath); // image.jpg
  //             const baseName = path.parse(imageName).name;

  //             const destImagePath = path.join(VERIFIED_DIR, imageName);
  //             const destPromptPath = path.join(VERIFIED_DIR, `${baseName}_prompt.txt`);
  //             const destBBoxPath = path.join(VERIFIED_DIR, `${baseName}_bbox.txt`);

  //             const unverifiedDir = path.dirname(srcPath);
  //             const originalPromptPath = path.join(unverifiedDir, `${baseName}_prompt.txt`);
  //             const originalBBoxPath = path.join(unverifiedDir, `${baseName}_bbox.txt`);

  //             // Move image
  //             if (fs.existsSync(srcPath)) {
  //                 fs.renameSync(srcPath, destImagePath);
  //             } else {
  //                 console.warn("Image not found:", srcPath);
  //                 return;
  //             }

  //             // Write prompt & bbox in verified dir
  //             fs.writeFileSync(destPromptPath, prompt);
  //             const { x, y, width, height } = boundingBox;
  //             fs.writeFileSync(destBBoxPath, `${x},${y},${width},${height}`);

  //             // Delete original prompt & bbox files
  //             if (fs.existsSync(originalPromptPath)) {
  //                 fs.unlinkSync(originalPromptPath);
  //             }

  //             if (fs.existsSync(originalBBoxPath)) {
  //                 fs.unlinkSync(originalBBoxPath);
  //             }

  //             console.log(`Verified and cleaned: ${imageName}`);
  //         });

  //         res.json({
  //             success: true,
  //             message: "Verified images saved and unverified files cleaned."
  //         });
  //     } catch (error) {
  //         console.error("Error saving verified data:", error);
  //         res.status(500).json({
  //             success: false,

  //             message: "Failed to store verified images",
  //             error: error.toString()
  //         });
  //     }
  // });
  mainRouter.post('/verified', async (req, res) => {
    const verifiedImages = req.body;

    if (!Array.isArray(verifiedImages) || verifiedImages.length === 0) {
      return res.status(400).json({
        success: false,
        message: "Send an array of verified image data."
      });
    }

    try {
      for (const item of verifiedImages) {
        const { imagePath, boundingBox, prompt } = item;
        console.log(imagePath, boundingBox, prompt);

        if (!imagePath || !boundingBox || !prompt) {
          console.warn("Skipping item due to missing data:", item);
          continue;
        }

        // Absolute paths
        const unverifiedDir = path.join(__dirname, '..', 'public');
        const srcPath = path.join(unverifiedDir, imagePath);
        const imageName = path.basename(srcPath);
        const baseName = path.parse(imageName).name;

        // Sanitize and create prompt folder
        const promptFolderName = prompt.trim().replace(/[<>:"/\\|?*\x00-\x1F]/g, '_');
        const promptDir = path.join(__dirname, 'verified', promptFolderName);

        // New subfolders
        const imagesDir = path.join(promptDir, 'images');
        const labelsDir = path.join(promptDir, 'labels');

        if (!fs.existsSync(imagesDir)) fs.mkdirSync(imagesDir, { recursive: true });
        if (!fs.existsSync(labelsDir)) fs.mkdirSync(labelsDir, { recursive: true });

        // Updated paths
        const destImagePath = path.join(imagesDir, imageName);
        const destBBoxPath = path.join(labelsDir, `${baseName}.txt`);

        const originalPromptPath = path.join(unverifiedDir, 'unverified', `${baseName}_prompt.txt`);
        const originalBBoxPath = path.join(unverifiedDir, 'unverified', `${baseName}_bbox.txt`);

        // Move image
        if (fs.existsSync(srcPath)) {
          fs.renameSync(srcPath, destImagePath);
        } else {
          console.warn("Image not found:", srcPath);
          continue;
        }

        // Save bbox
        const { x, y, width, height } = boundingBox;
        fs.writeFileSync(destBBoxPath, `${x},${y},${width},${height}`);

        // Delete originals
        if (fs.existsSync(originalPromptPath)) fs.unlinkSync(originalPromptPath);
        if (fs.existsSync(originalBBoxPath)) fs.unlinkSync(originalBBoxPath);

        console.log(`Verified and saved → ${promptFolderName}/images/${imageName} and labels/${baseName}_bbox.txt`);
      }


      // Triggering the python code to re-train the model if the counter of images has reached the desired number
      try {

        const response = await axios.get('http://localhost:3000/autotrain');

        console.log(response.data.status);
        if (response.data.status !== 'success') {
          console.warn("Flask training did not complete successfully:", response.data.message);
        }

      } catch (error) {
        console.error('Error calling Flask:', error.message);

      }
       

      res.json({
        success: true,
        message: "Verified images saved in prompt folders with images and labels subfolders."
      });

    } catch (error) {
      console.error("Error saving verified data:", error);
      res.status(500).json({
        success: false,
        message: "Failed to store verified images",
        error: error.toString()
      });
    }
  });

  const getTimeStampFilename = () => Date.now().toString();

  // MJPEG frame capture logic
  function captureFramesFromStream(url, frameLimit = 5, prompt = '') {
    return new Promise((resolve, reject) => {
      let buffer = Buffer.alloc(0);
      let frameCount = 0;
      const savedFrames = [];

      const request = http.get(url, (res) => {
        res.on('data', (chunk) => {
          buffer = Buffer.concat([buffer, chunk]);
          const start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
          const end = buffer.indexOf(Buffer.from([0xff, 0xd9]));

          if (start !== -1 && end !== -1 && end > start) {
            const jpgBuffer = buffer.slice(start, end + 2);
            buffer = buffer.slice(end + 2);

            const timestamp = getTimeStampFilename();
            const imageFilename = `${timestamp}.jpg`;
            const imagePath = path.join(FRAMES_DIR, imageFilename);
            fs.writeFileSync(imagePath, jpgBuffer);

            const promptFilename = `${timestamp}.txt`;
            const promptPath = path.join(FRAMES_DIR, promptFilename);
            fs.writeFileSync(promptPath, prompt);

            savedFrames.push({ image: imageFilename, promptFile: promptFilename });
            console.log(`Saved frame: ${imageFilename}`);

            if (++frameCount >= frameLimit) {
              res.destroy();
              resolve(savedFrames);
            }
          }
        });

        res.on('end', () => resolve(savedFrames));
        res.on('error', reject);
      });

      request.on('error', reject);
    });
  }

  // GET /capture_frames
  mainRouter.get('/capture_frames', async (req, res) => {
    const numFrames = parseInt(req.query.count) || 20;
    const prompt = req.query.prompt?.trim();

    if (!prompt) {
      return res.status(400).json({ status: 'error', message: 'Missing required prompt.' });
    }

    try {
      const frames = await captureFramesFromStream(MJPEG_URL, numFrames, prompt);
      res.json({ status: 'success', message: `Captured ${frames.length} frame(s)`, files: frames });
    } catch (err) {
      console.error('Capture error:', err);
      res.status(500).json({ status: 'error', message: 'Failed to capture frames', error: err.toString() });
    }
  });

  // GET /unverified_images
  mainRouter.get('/unverified_images', (req, res) => {
    try {
      if (!fs.existsSync(UNVERIFIED_DIR)) {
        return res.status(404).json({ status: 'error', message: 'Unverified folder not found.' });
      }

      const files = fs.readdirSync(UNVERIFIED_DIR);
      const imageFiles = files.filter(f => f.endsWith('.jpg'));

      const results = imageFiles.flatMap(imageFile => {
        const baseName = path.parse(imageFile).name;
        const promptPath = path.join(UNVERIFIED_DIR, `${baseName}_prompt.txt`);
        const bboxPath = path.join(UNVERIFIED_DIR, `${baseName}_bbox.txt`);

        if (!fs.existsSync(promptPath) || !fs.existsSync(bboxPath)) {
          console.warn(`Missing prompt/bbox for ${imageFile}, skipping.`);
          return [];
        }

        const prompt = fs.readFileSync(promptPath, 'utf-8').trim();
        const [x, y, width, height] = fs.readFileSync(bboxPath, 'utf-8').trim().split(/[\s,]+/).map(Number);

        return {
          imagePath: `unverified/${imageFile}`,
          boundingBox: { x, y, width, height },
          prompt
        };
      });

      res.json(results);
    } catch (err) {
      console.error('Error reading unverified images:', err);
      res.status(500).json({ status: 'error', message: 'Failed to load images', error: err.toString() });
    }
  });

  // POST /verified
  mainRouter.post('/verified', (req, res) => {
    const verifiedImages = req.body;

    if (!Array.isArray(verifiedImages) || verifiedImages.length === 0) {
      return res.status(400).json({ success: false, message: 'Send a non-empty array.' });
    }

    try {
      verifiedImages.forEach(item => {
        const { imagePath, boundingBox, prompt } = item;
        if (!imagePath || !boundingBox || !prompt) {
          console.warn('Invalid entry:', item);
          return;
        }

        const srcPath = path.join(__dirname, '..', 'public', imagePath);
        const imageName = path.basename(srcPath);
        const baseName = path.parse(imageName).name;

        const promptFolder = prompt.trim().replace(/[<>:"/\\|?*\x00-\x1F]/g, '_');
        const promptDir = path.join(VERIFIED_DIR, promptFolder);
        if (!fs.existsSync(promptDir)) fs.mkdirSync(promptDir, { recursive: true });

        const destImagePath = path.join(promptDir, imageName);
        const destBBoxPath = path.join(promptDir, `${baseName}_bbox.txt`);
        const originalPromptPath = path.join(UNVERIFIED_DIR, `${baseName}_prompt.txt`);
        const originalBBoxPath = path.join(UNVERIFIED_DIR, `${baseName}_bbox.txt`);

        if (fs.existsSync(srcPath)) fs.renameSync(srcPath, destImagePath);
        fs.writeFileSync(destBBoxPath, `${boundingBox.x},${boundingBox.y},${boundingBox.width},${boundingBox.height}`);

        if (fs.existsSync(originalPromptPath)) fs.unlinkSync(originalPromptPath);
        if (fs.existsSync(originalBBoxPath)) fs.unlinkSync(originalBBoxPath);

        console.log(`[VERIFY] ${imageName} -> ${promptFolder}/`);
      });

      res.json({ success: true, message: 'Images moved and bbox stored in prompt folders.' });
    } catch (err) {
      console.error('Verify error:', err);
      res.status(500).json({ success: false, message: 'Failed to process verified images.', error: err.toString() });
    }
  });

  // POST /deleteImage
  mainRouter.post('/deleteImage', (req, res) => {
    console.log("Inside the delete path");
    const imageEntries = req.body;

    if (!Array.isArray(imageEntries) || imageEntries.length === 0) {
      return res.status(400).json({ success: false, message: 'Invalid input format.' });
    }

    const deleted = [], errors = [];

    imageEntries.forEach(({ imagePath }) => {
      try {
        const fullImagePath = path.join(__dirname, '..', 'public', imagePath);
        const baseName = path.parse(fullImagePath).name;
        const dir = path.dirname(fullImagePath);

        [fullImagePath, `${baseName}_prompt.txt`, `${baseName}_bbox.txt`].forEach((file, idx) => {
          const filePath = idx === 0 ? file : path.join(dir, file);
          if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            if (idx === 0) deleted.push(imagePath);
          }
        });
      } catch (err) {
        errors.push({ imagePath, error: err.toString() });
      }
    });

    res.status(errors.length ? 207 : 200).json({
      success: errors.length === 0,
      deleted,
      errors,
      message: errors.length ? 'Some deletions failed.' : 'All files deleted successfully.'
    });
  });

  // Static preview of frames
  mainRouter.use('/frames', express.static(FRAMES_DIR));

  module.exports = mainRouter;
})