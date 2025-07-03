import { useEffect, useRef, useState } from "react";
import { Rnd } from "react-rnd";
import axios from "axios";
import Navbar from "../components/Navbar";

function VerifyImages() {
  const [data, setData] = useState([]);
  const [verifiedImages, setVerifiedImages] = useState([]);
  const [deletedImages, setDeletedImages] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [box, setBox] = useState(null);
  const imageRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [lastImageReviewed, setLastImageReviewed] = useState(false);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:3001/unverified_images");
        setData(res.data);
        if (res.data.length > 0) setCurrentIdx(0);
      } catch (err) {
        console.error("Error fetching image data:", err);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    const updateSize = () => {
      if (imageRef.current) {
        const rect = imageRef.current.getBoundingClientRect();
        setImageSize({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    if (data.length > 0 && data[currentIdx]) {
      setBox({ ...data[currentIdx].boundingBox });
    }
  }, [currentIdx, data]);

  // useEffect(() => {
  //   const handleBeforeUnload = () => {
  //     if (verifiedImages.length) {
  //       const blob = new Blob([JSON.stringify(verifiedImages)], { type: "application/json" });
  //       navigator.sendBeacon("http://localhost:3001/verified", blob);
  //     }
  //     if (deletedImages.length) {
  //       const blob = new Blob([JSON.stringify(deletedImages)], { type: "application/json" });
  //       navigator.sendBeacon("http://localhost:3001/deleteImage", blob);
  //     }
  //   };

  //   window.addEventListener("beforeunload", handleBeforeUnload);
  //   return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  // }, [verifiedImages, deletedImages]);

  const uploadPendingData = async () => {
    try {
      if (!verifiedImages.length && !deletedImages.length) { alert('No updates to post'); return; }
      if (verifiedImages.length) {
        await axios.post("http://localhost:3001/verified", verifiedImages);
        console.log("Verified images uploaded.");
      }
      if (deletedImages.length) {
        await axios.post("http://localhost:3001/deleteImage", { imagePath: deletedImages.map(img => img.imagePath), });
        console.log("Deleted images uploaded.");

      }
      alert(`${verifiedImages.length} images uploaded, ${deletedImages.length} images deleted`);
      setVerifiedImages([]);
      setDeletedImages([]);


    } catch (err) {
      console.error("Failed to upload verified or deleted images:", err);
    }
  };

  const moveToNext = () => {
    setCurrentIdx((prev) => {
      if (prev < data.length - 1) return prev + 1;
      return prev; // stay on last image
    });
  };

  const handleSaveAndNext = async () => {
    const item = data[currentIdx];
    const updatedItem = {
      prompt: item.prompt,
      imagePath: item.imagePath,
      boundingBox: box,
    };

    const isLast = currentIdx === data.length - 1;

    // Save the image
    setVerifiedImages((prev) => [...prev, updatedItem]);

    if (isLast) {
      alert("All images reviewed. Post the updates for !");
      setLastImageReviewed(true);
    } else {
      // Just move to next
      moveToNext();
    }
  };

  const handleDelete = () => {
    const item = data[currentIdx];
    setDeletedImages((prev) => [...prev, { imagePath: item.imagePath }]);

    const isLast = currentIdx === data.length - 1;
    if (isLast) {
      setLastImageReviewed(true);
      alert("All images reviewed. Please post the updates.");
    } else {
      moveToNext();
    }
  };

  const handlePostUpdates = async () => {
    await uploadPendingData();
  };

  if (data.length === 0 || !data[currentIdx] || !box) {
    return <>
      <Navbar heading="Verify Images" />
      <div className="text-center mt-10">No unverified images.</div>;
    </>
  }

  const current = data[currentIdx];
  const imageUrl = `http://localhost:3001/static/${current.imagePath}`;

  const absoluteBox = {
    x: Math.min(box.x * imageSize.width, imageSize.width),
    y: Math.min(box.y * imageSize.height, imageSize.height),
    width: Math.min(box.width * imageSize.width, imageSize.width),
    height: Math.min(box.height * imageSize.height, imageSize.height),
  };

  return (
    <>
      <Navbar heading="Verify Images" />
      <div className="outer-box p-6 font-sans">
        <div className="verifier-container mx-auto w-[90vw] max-w-[800px] bg-white rounded-lg shadow-lg p-6">
          <div className="header-panel mb-4">
            <p className="text-lg font-semibold text-gray-700 mb-1">
              Prompt: <span className="font-normal">{current.prompt}</span>
            </p>
            <p className="text-sm text-gray-500">
              Image {currentIdx + 1} of {data.length}
            </p>
            <p className="text-sm text-gray-500">
              Note : Please post updates after verifying the images
            </p>
          </div>

          <div className="image-wrapper relative inline-block w-full max-w-3xl mx-auto aspect-video bg-gray-100 rounded-md overflow-hidden border border-gray-300">
            <img
              ref={imageRef}
              src={imageUrl}
              alt="To verify"
              className="w-full h-full object-contain"
              onLoad={() => {
                const rect = imageRef.current.getBoundingClientRect();
                setImageSize({ width: rect.width, height: rect.height });
              }}
            />

            {imageSize.width > 0 && (
              <Rnd
                size={{ width: absoluteBox.width, height: absoluteBox.height }}
                position={{ x: absoluteBox.x, y: absoluteBox.y }}
                onDragStop={(e, d) => {
                  const clampedX = Math.max(0, Math.min(d.x, imageSize.width - absoluteBox.width));
                  const clampedY = Math.max(0, Math.min(d.y, imageSize.height - absoluteBox.height));
                  setBox((prev) => ({
                    ...prev,
                    x: clampedX / imageSize.width,
                    y: clampedY / imageSize.height,
                  }));
                }}
                onResizeStop={(e, direction, ref, delta, position) => {
                  const rawWidth = parseInt(ref.style.width);
                  const rawHeight = parseInt(ref.style.height);
                  const clampedWidth = Math.max(1, Math.min(rawWidth, imageSize.width - position.x));
                  const clampedHeight = Math.max(1, Math.min(rawHeight, imageSize.height - position.y));

                  const clampedX = Math.max(0, Math.min(position.x, imageSize.width - clampedWidth));
                  const clampedY = Math.max(0, Math.min(position.y, imageSize.height - clampedHeight));

                  setBox({
                    width: clampedWidth / imageSize.width,
                    height: clampedHeight / imageSize.height,
                    x: clampedX / imageSize.width,
                    y: clampedY / imageSize.height,
                  });
                }}
                bounds={imageRef.current ? imageRef.current : "parent"}
                style={{
                  border: "2px solid red",
                  backgroundColor: "rgba(255, 0, 0, 0.1)",
                  position: "absolute",
                }}
              />
            )}
          </div>

          <div className="button-bar mt-6 flex justify-center gap-4">
            <button type="button" onClick={handleSaveAndNext} disabled={lastImageReviewed}>
              Next
            </button>
            <button type="button" onClick={handleDelete} disabled={lastImageReviewed}>
              Delete
            </button>
            <button type="button" onClick={handlePostUpdates}>
              Post Updates
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

export default VerifyImages;