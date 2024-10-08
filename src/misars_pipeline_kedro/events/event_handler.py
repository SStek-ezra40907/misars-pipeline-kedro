def handle_event(event):
    if event == Event.START_TASK:
        print("Starting task...")
        # 可以放置調用任務的邏輯
    elif event == Event.STOP_TASK:
        print("Stopping task...")
        # 可以放置停止任務的邏輯