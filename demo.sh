#!/bin/bash
set -e

echo "======================================"
echo "    JAGRIT LIVE DEMO SCRIPT           "
echo "======================================"

echo "1. Checking environment..."
if [ -z "$GEMINI_API_KEY" ]; then
  echo "WARNING: GEMINI_API_KEY is not set. Hindi translations may fail or be mocked."
else
  echo "GEMINI_API_KEY is configured."
fi

echo -e "\n2. Starting Docker Compose stack..."
docker-compose up -d --build

echo -e "\n3. Waiting for services to become healthy (sleep 15s)..."
sleep 15

echo -e "\n4. Starting Kafka Event Replay (simulating live production traffic)..."
# Start the producer in the background
cd ml-service && python streaming/producer.py --replay-speed 1000 &
PRODUCER_PID=$!
cd ..

echo -e "\n======================================"
echo " SYSTEM IS UP AND RUNNING!"
echo "======================================"
echo ""
echo "Follow this Demo Script:"
echo "1. Open http://localhost:3000 in your browser."
echo "2. Click 'Login' and register a new user."
echo "   - Keep MIND User ID as 'U13740' to see ML rankings for a real active user."
echo "   - Or change to 'NEW_USER_999' to see the cold-start fallback logic."
echo "3. Go to the 'For You' page and view the ranked editorial feed."
echo "4. Toggle the language to 'Hindi' in the navbar."
echo "   - Notice the first load takes a moment (Gemini API), but subsequent reloads are instant (MongoDB Cache)."
echo "5. Click the [DEV] button on any card to reveal the XGBoost ranking score."
echo "6. Save an article, navigate to Preferences, and update your categories."
echo ""
echo "Press Ctrl+C to tear down the environment."

# Wait for user interrupt
trap "echo 'Tearing down...'; kill $PRODUCER_PID; docker-compose down; exit 0" INT
wait
