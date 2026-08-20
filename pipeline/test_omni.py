import asyncio
import websockets

async def test_stream():
    # Swap out the path below to match your backend wrapper
    url = "ws://localhost:8080/ws"
    async with websockets.connect(url) as ws:
        print("Successfully connected to OmniVoice WebSocket!")
        # Example JSON configuration message to start text injection
        await ws.send('{"text": "Hello world "}')
        response = await ws.recv()
        print("Received chunk:", response[:50]) # Truncated raw audio bytes

asyncio.run(test_stream())