const express = require('express');
const fs = require('fs');
const path = require('path');
const http = require('http');

const mainRouter=new express.Router;

const MJPEG_URL = 'http://localhost:3000/video';

// Directory for saving frames and prompts (Its name is currently just frames.)
const FRAMES_DIR = path.join(__dirname, 'frames');
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
//THis is kinda incomplete.
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

    imagePath.forEach((relativePath) => {
        try {
            const fullImagePath = path.join(__dirname, relativePath);
            const baseName = path.parse(fullImagePath).name;
            const dir = path.dirname(fullImagePath);

            const promptPath = path.join(dir, `${baseName}_prompt.txt`);
            const bboxPath = path.join(dir, `${baseName}_bbox.txt`);

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
if (!fs.existsSync(VERIFIED_DIR)) {
    fs.mkdirSync(VERIFIED_DIR, { recursive: true });
}


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
mainRouter.post('/verified', (req, res) => {
    const verifiedImages = req.body;

    if (!Array.isArray(verifiedImages) || verifiedImages.length === 0) {
        return res.status(400).json({
            success: false,
            message: "Send an array of verified image data."
        });
    }

    try {
        verifiedImages.forEach(item => {
            const { imagePath, boundingBox, prompt } = item;
            console.log(imagePath, boundingBox, prompt);

            if (!imagePath || !boundingBox || !prompt) {
                console.warn("Skipping item due to missing data:", item);
                return;
            }

            // Absolute paths
            const unverifiedDir = path.join(__dirname, '..', 'public');
            const srcPath = path.join(unverifiedDir, imagePath); // e.g., /public/unverified/xyz.jpg
            const imageName = path.basename(srcPath);
            const baseName = path.parse(imageName).name;

            const verifiedDir = path.join(__dirname, 'verified'); // backend/routes/verified
            const destImagePath = path.join(verifiedDir, imageName);
            const destPromptPath = path.join(verifiedDir, `${baseName}_prompt.txt`);
            const destBBoxPath = path.join(verifiedDir, `${baseName}_bbox.txt`);

            const originalPromptPath = path.join(unverifiedDir, 'unverified', `${baseName}_prompt.txt`);
            const originalBBoxPath = path.join(unverifiedDir, 'unverified', `${baseName}_bbox.txt`);

            // Move image
            if (fs.existsSync(srcPath)) {
                fs.renameSync(srcPath, destImagePath);
            } else {
                console.warn("Image not found:", srcPath);
                return;
            }

            // Write prompt and bbox
            fs.writeFileSync(destPromptPath, prompt);
            const { x, y, width, height } = boundingBox;
            fs.writeFileSync(destBBoxPath, `${x},${y},${width},${height}`);

            // Delete original .txt files
            if (fs.existsSync(originalPromptPath)) fs.unlinkSync(originalPromptPath);
            if (fs.existsSync(originalBBoxPath)) fs.unlinkSync(originalBBoxPath);

            console.log(`Verified and cleaned: ${imageName}`);
        });

        res.json({
            success: true,
            message: "Verified images saved and unverified files cleaned."
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

mainRouter.use('/frames', express.static(FRAMES_DIR));

module.exports=mainRouter;