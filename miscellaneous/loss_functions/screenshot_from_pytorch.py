import os

from playwright.sync_api import sync_playwright

# Directory to save the captured images
output_dir = "katex_images"
os.makedirs(output_dir, exist_ok=True)

# URLs to visit
urls = {
    "BCEWithLogitsLoss": "https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html",
    "CrossEntropyLoss": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"
}

# Start Playwright and navigate to the URLs
with sync_playwright() as p:
    # Launch a headless browser
    browser = p.chromium.launch(headless=True)

    # Iterate through both URLs
    for loss_name, url in urls.items():
        # Create a new page with a large viewport and high deviceScaleFactor (DPI)
        page = browser.new_page(
            viewport={"width": 8640, "height": 8640},  # 8K resolution (8640x8640)
            device_scale_factor=16  # Increase the device scale factor for high-resolution rendering
        )

        # Go to the URL
        page.goto(url)

        # Wait for the page to load completely
        page.wait_for_load_state('networkidle')

        # Evaluate JavaScript to remove certain elements as required
        page.evaluate("""
            const mordElements = document.querySelectorAll('.mord');
            mordElements.forEach(mord => {
                const spanChild = mord.querySelector('span.mord.mathnormal');
                if (spanChild && spanChild.textContent.trim() === 'w') {
                    mord.remove();  // Remove the parent element
                }
            });

            // Remove the 5th element with class="mpunct"
            const mpunctElements = document.querySelectorAll('.mpunct');
            if (mpunctElements.length >= 5) {
                mpunctElements[4].remove();  // Index 4 is the 5th element (0-based index)
            }
        """)

        # Find all elements with class 'katex-html'
        katex_elements = page.query_selector_all(".katex-html")

        # Capture the relevant element as PNG
        if loss_name == "BCEWithLogitsLoss":
            # Capture the first KaTeX element on the BCEWithLogitsLoss page
            if katex_elements:
                file_name = os.path.join(output_dir, f"{loss_name}.png")
                katex_elements[0].screenshot(path=file_name)
                print(f"Saved: {file_name}")
        elif loss_name == "CrossEntropyLoss":
            # Capture the 15th KaTeX element on the CrossEntropyLoss page
            if len(katex_elements) >= 15:
                file_name = os.path.join(output_dir, f"{loss_name}.png")
                katex_elements[14].screenshot(path=file_name)  # Index 14 is the 15th element (0-based index)
                print(f"Saved: {file_name}")

    # Close the browser
    browser.close()
