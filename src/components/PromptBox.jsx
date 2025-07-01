import * as React from 'react';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';

export default function PromptBox() {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const formJson = Object.fromEntries(formData.entries());
    const prompt = formJson.prompt.trim();
    const count = formJson.count || 5;

    if (!prompt) {
      alert("Prompt is required.");
      return;
    }

    handleClose();
    alert('Prompt submitted successfully.');

    try {
      const url = `http://localhost:3001/capture_frames?prompt=${encodeURIComponent(prompt)}&count=${count}`;
      const response = await fetch(url, { method: 'GET' });

      if (!response.ok) {
        console.error('Server error:', await response.text());
        return;
      }

      const result = await response.json();
      console.log('Captured frames:', result.files);
      alert(`Captured ${result.files.length} frame(s).`);
    } catch (error) {
      console.error('Fetch error:', error);
      alert('Failed to capture frames.');
    }
  };

  return (
    <React.Fragment>
      <button
        onClick={handleClickOpen}
        className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
        aria-label="Add new event"
      >
        Add Event
      </button>
      <Dialog open={open} onClose={handleClose}>

        <form onSubmit={handleSubmit}>
          <DialogContent>
            <DialogContentText>
              <span> Enter the name of the event to be added </span>
              <span>(Note: Be precise with the event name for better results)</span>
            </DialogContentText>
            <TextField
              autoFocus
              required
              margin="dense"
              id="prompt"
              name="prompt"
              label="Prompt"
              type="text"
              fullWidth
              variant="standard"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose}>Cancel</Button>
            <Button type="submit">Submit</Button>
          </DialogActions>
        </form>
      </Dialog>
    </React.Fragment>
  );
}