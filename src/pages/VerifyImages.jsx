import { useEffect, useRef, useState } from "react";
import { Rnd } from "react-rnd";
import axios from "axios";
import Navbar from "../components/Navbar";

function VerifyImages() {
  const [data, setData] = useState([
    {
      imagePath: "/unverified/0001.jpeg",
      prompt: "Sanitizing hands before patient contact",
      boundingBox: { x: 0.2, y: 0.25, width: 0.3, height: 0.3 }
    },
    {
      imagePath: "/unverified/0002.jpeg",
      prompt: "Nurse wearing gloves while dressing wound",
      boundingBox: { x: 0.1, y: 0.15, width: 0.25, height: 0.35 }
    }
  ]);

  const [verifiedImages, setVerifiedImages] = useState([]);
  const [deletedImages, setdeletedImages] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [box, setBox] = useState({ ...data[0].boundingBox });
  const imageRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:3001/unverified_images");
        setData(res.data);
        console.log(res.data);
        if (res.data.length > 0) setBox({ ...res.data[0].boundingBox });
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
    const handleBeforeUnload = () => {
      if (verifiedImages.length) {
        const blob = new Blob([JSON.stringify(verifiedImages)], { type: "application/json" });
        navigator.sendBeacon("http://localhost:3001/verified", blob);
      }
      if (deletedImages.length) {
        const blob = new Blob([JSON.stringify(deletedImages)], { type: "application/json" });
        navigator.sendBeacon("http://localhost:3001/deleteImage", blob);
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [verifiedImages, deletedImages]);

  const uploadPendingData = async () => {
    try {
      if (verifiedImages.length) {
        await axios.post("http://localhost:3001/verified", verifiedImages);
        console.log("Verified images uploaded.");
        setVerifiedImages([]);
      }
      // deleting the image logic may be changed as per the backend requirements 
      if (deletedImages.length) {
        await axios.post("http://localhost:3001/deleteImage", imagePath);
        console.log("Deleted images uploaded.");
        setdeletedImages([]);
      }
    } catch (err) {
      console.error("Failed to upload verified or deleted images:", err);
    }
  };

  const handleSaveAndNext = () => {
    const item = data[currentIdx];
    const updatedItem = {
      prompt: item.prompt,
      imagePath: item.imagePath,
      boundingBox: box,
    };
    setVerifiedImages((prev) => [...prev, updatedItem]);
    moveToNext();
  };


  const handleDelete = () => {
    const item = data[currentIdx];
    setdeletedImages((prev) => [...prev, { imagePath: item.imagePath }]);
    moveToNext();
  };

  const handleSkip = () => {
    moveToNext();
  };

  const handlePostUpdates = () => {
    uploadPendingData();
  }

  const moveToNext = async () => {
    const nextIdx = currentIdx + 1;
    if (nextIdx < data.length) {
      setCurrentIdx(nextIdx);
      setBox({ ...data[nextIdx].boundingBox });
    } else {
      alert("All images reviewed!");
      await uploadPendingData();
      if (window.opener) {
        window.opener.focus();
      }
    }
  };

  if (data.length === 0 || !box) return <div>Loading...</div>;

  const current = data[currentIdx];
  const imageUrl = `${current.imagePath}`;

  const absoluteBox = {
    x: box.x * imageSize.width,
    y: box.y * imageSize.height,
    width: box.width * imageSize.width,
    height: box.height * imageSize.height
  };

  return (
    <>
      <Navbar heading="Verify Images" />
      <div className="outer-box p-6 font-sans">


        <div className="verifier-container mx-auto w-[90vw] max-w-[800px] bg-white rounded-lg shadow-lg p-6">

          {/* Header */}
          <div className="header-panel mb-4">
            <p className="text-lg font-semibold text-gray-700 mb-1">
              Prompt: <span className="font-normal">{current.prompt}</span>
            </p>
            <p className="text-sm text-gray-500">
              Image {currentIdx + 1} of {data.length}
            </p>
          </div>

          {/* Image Wrapper */}
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
                  const newX = Math.max(0, Math.min(d.x, imageSize.width - absoluteBox.width));
                  const newY = Math.max(0, Math.min(d.y, imageSize.height - absoluteBox.height));
                  setBox(prev => ({
                    ...prev,
                    x: newX / imageSize.width,
                    y: newY / imageSize.height
                  }));
                }}
                onResizeStop={(e, direction, ref, delta, position) => {
                  const newWidth = Math.min(parseInt(ref.style.width), imageSize.width - position.x);
                  const newHeight = Math.min(parseInt(ref.style.height), imageSize.height - position.y);
                  const newX = Math.max(0, Math.min(position.x, imageSize.width - newWidth));
                  const newY = Math.max(0, Math.min(position.y, imageSize.height - newHeight));
                  setBox({
                    width: newWidth / imageSize.width,
                    height: newHeight / imageSize.height,
                    x: newX / imageSize.width,
                    y: newY / imageSize.height
                  });
                }}
                bounds={imageRef.current ? imageRef.current : "parent"}
                style={{
                  border: "2px solid red",
                  backgroundColor: "rgba(255, 0, 0, 0.1)",
                  position: "absolute"
                }}
              />
            )}
          </div>

          {/* Button Bar */}
          <div className="button-bar mt-6 flex justify-center gap-4">
            <button onClick={handleSaveAndNext}>Save</button>
            {/* <button onClick={handleSkip}>Skip</button> */}
            <button onClick={handleDelete}>Delete</button>
            <button onClick={handlePostUpdates}>Post Updates</button>
          </div>
        </div>
      </div>
    </>
  );
}

export default VerifyImages;