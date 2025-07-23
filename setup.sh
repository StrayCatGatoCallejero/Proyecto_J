#!/bin/bash
# Setup script for Render deployment

# Install system dependencies for Kaleido
apt-get update
apt-get install -y wget gnupg

# Install Chrome for Kaleido
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
apt-get update
apt-get install -y google-chrome-stable

# Set environment variables
export DISPLAY=:99
export CHROME_BIN=/usr/bin/google-chrome

echo "Setup completed successfully!" 