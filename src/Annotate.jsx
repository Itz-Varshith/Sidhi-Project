import React from "react";
import { useState, useEffect, useRef } from "react";
import { Stage, Layer, Image, Rect, Text } from "react-konva";

const Annotate = () => {
    const [image, setImage] = useState(null);
    const [annotations, setAnnotations] = useState([]);
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentRect, setCurrentRect] = useState(null);
    const [startPos, setStartPos] = useState({ x: 0, y: 0 });
    const [selectedIndex, setSelectedIndex] = useState(null);
    const [showInput, setShowInput] = useState(false);
    const [inputPos, setInputPos] = useState({ x: 0, y: 0 });
    const [inputText, setInputText] = useState('');
    const stageRef = useRef(null);
    const inputRef = useRef(null);
  
    useEffect(() => {
      const img = new window.Image();
      img.src = 'https://imgs.search.brave.com/gxcCQAlsUVYkRiLFjl6g9r-QliohhKTb90wuCuUSx3A/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/cHJlbWl1bS1waG90/by9mYXN0LXN1cGVy/Y2Fycy1waG90b181/NTE3MDctMjE1NS5q/cGc_c2VtdD1haXNf/aHlicmlkJnc9NzQw';
      img.crossOrigin = 'Anonymous';
      img.onload = () => setImage(img);
    }, []);
  
    useEffect(() => {
      if (showInput && inputRef.current) {
        inputRef.current.focus();
      }
    }, [showInput]);
  
    const handleMouseDown = (e) => {
      if (!isDrawing) {
        const pos = e.target.getStage().getPointerPosition();
        setStartPos(pos);
        setIsDrawing(true);
        setSelectedIndex(null);
        setShowInput(false);
      }
    };
  
    const handleMouseMove = (e) => {
      if (isDrawing) {
        const pos = e.target.getStage().getPointerPosition();
        const newRect = {
          x: Math.min(startPos.x, pos.x),
          y: Math.min(startPos.y, pos.y),
          width: Math.abs(pos.x - startPos.x),
          height: Math.abs(pos.y - startPos.y),
          stroke: 'red',
          strokeWidth: 2,
        };
        setCurrentRect(newRect);
      }
    };
  
    const handleMouseUp = () => {
      if (isDrawing) {
        setIsDrawing(false);
        if (currentRect) {
          const stage = stageRef.current;
          const stagePos = stage.getAbsolutePosition();
          setInputPos({
            x: stagePos.x + currentRect.x + currentRect.width + 10,
            y: stagePos.y + currentRect.y + currentRect.height + 10,
          });
          setShowInput(true);
          setInputText('');
        }
      }
    };
  
    const handleRectClick = (index) => {
      setSelectedIndex(index === selectedIndex ? null : index);
      setShowInput(false);
    };
  
    const handleInputSubmit = () => {
      if (currentRect) {
        const text = inputText.trim() || 'Unnamed';
        setAnnotations([...annotations, { rect: currentRect, text }]);
        setCurrentRect(null);
        setShowInput(false);
        setInputText('');
      }
    };
  
    const handleInputKeyDown = (e) => {
      if (e.key === 'Enter') {
        handleInputSubmit();
      }
    };
  
    const handleSubmitImage = () => {
      const dataURL = stageRef.current.toDataURL();
      const link = document.createElement('a');
      link.href = dataURL;
      link.download = 'annotated-image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
  
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
        This is the MVP verison of the Annotate page(Only the features work and also the submit button downloads the image right now instead of sending the image to the backend.).
        <h1 className="text-2xl font-bold text-gray-800 mb-4">Drag to draw rectangles</h1>
        <div className="relative">
          <Stage
            width={600}
            height={400}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            ref={stageRef}
            className="border border-gray-300 shadow-lg"
          >
            <Layer>
              {image && <Image image={image} width={600} height={400} />}
              {annotations.map((anno, i) => (
                <>
                  <Rect
                    key={`rect-${i}`}
                    {...anno.rect}
                    onClick={() => handleRectClick(i)}
                    stroke={selectedIndex === i ? 'blue' : 'red'}
                  />
                  {selectedIndex === i && (
                    <Rect
                      key={`highlight-${i}`}
                      x={anno.rect.x + anno.rect.width / 2 - (anno.text.length * 8) / 2}
                      y={anno.rect.y + anno.rect.height}
                      width={anno.text.length * 8}
                      height={20}
                      fill="yellow"
                      opacity={0.5}
                    />
                  )}
                  <Text
                    key={`text-${i}`}
                    x={anno.rect.x + anno.rect.width / 2}
                    y={anno.rect.y + anno.rect.height + 5}
                    text={anno.text}
                    fontSize={16}
                    fill="red"
                    align="center"
                    offsetX={anno.text.length * 4}
                  />
                </>
              ))}
              {currentRect && <Rect {...currentRect} />}
            </Layer>
          </Stage>
          {showInput && (
            <div
              className="absolute flex items-center bg-white border border-gray-300 rounded-md shadow-md p-2"
              style={{ left: `${inputPos.x}px`, top: `${inputPos.y}px` }}
            >
              <input
                ref={inputRef}
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleInputKeyDown}
                className="px-2 py-1 border-none outline-none text-gray-700"
                placeholder="Enter name"
              />
              <button
                onClick={handleInputSubmit}
                className="ml-2 px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                OK
              </button>
            </div>
          )}
        </div>
        <button
          onClick={handleSubmitImage}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Submit Image
        </button>
      </div>
    );
};

export default Annotate;
