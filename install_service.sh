#!/bin/bash

# This script will install the systemd service.
# Run this script with sudo ON THE RASPBERRY PI ONLY.

SERVICE_NAME="yoga-judge.service"
SERVICE_TEMPLATE="./yoga-judge.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "==========================================="
echo " Installing Yoga Judge Systemd Service..."
echo "==========================================="

if [ "$EUID" -ne 0 ]; then 
  echo "Error: Please run this script with sudo (e.g., sudo ./install_service.sh)"
  exit 1
fi

if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo "Error: Could not find $SERVICE_TEMPLATE in the current directory."
    exit 1
fi

# Copy the service file to systemd directory
echo "Copying $SERVICE_TEMPLATE to $SYSTEMD_DIR/"
cp "$SERVICE_TEMPLATE" "$SYSTEMD_DIR/"

# Set correct permissions
chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME"

# Reload systemd daemon to recognize the new service
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service to start automatically on boot
echo "Enabling $SERVICE_NAME to start on boot..."
systemctl enable "$SERVICE_NAME"

echo ""
echo "==========================================="
echo " Installation Complete!"
echo "==========================================="
echo "You can now control the app using these commands:"
echo ""
echo "  sudo systemctl start $SERVICE_NAME"
echo "  sudo systemctl stop $SERVICE_NAME"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view live logs from the app, run:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo "==========================================="
