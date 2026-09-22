self.onmessage = async (event: MessageEvent<{ id: string; file: File }>) => {
  try {
    const bytes = await event.data.file.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hex = [...new Uint8Array(digest)]
      .map((v) => v.toString(16).padStart(2, "0"))
      .join("");
    postMessage({ id: event.data.id, sha256: hex });
  } catch (error) {
    postMessage({
      id: event.data.id,
      error: error instanceof Error ? error.message : "Hash failed",
    });
  }
};
