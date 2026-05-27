async function fileToRecord(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return { name: file.name, data_url: dataUrl };
}

async function filesToRecords(input) {
  const files = Array.from(input.files || []);
  return Promise.all(files.map(fileToRecord));
}

const localizationRecords = [];

function renderLocalizationMemory() {
  const count = document.getElementById("localization-count");
  const list = document.getElementById("localization-list");
  count.textContent = `已累積 ${localizationRecords.length} 張定位照片`;
  list.innerHTML = "";
  for (const [index, record] of localizationRecords.entries()) {
    const item = document.createElement("li");
    item.textContent = `${index + 1}. ${record.name}`;
    list.appendChild(item);
  }
}

async function optionalPassageRecord(id) {
  const input = document.getElementById(id);
  if (!input.files || input.files.length === 0) {
    return null;
  }
  return fileToRecord(input.files[0]);
}

document.getElementById("localization-images").addEventListener("change", async (event) => {
  const records = await filesToRecords(event.target);
  localizationRecords.push(...records);
  event.target.value = "";
  renderLocalizationMemory();
});

document.getElementById("clear-localization-images").addEventListener("click", () => {
  localizationRecords.length = 0;
  renderLocalizationMemory();
});

document.getElementById("guidance-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = document.getElementById("message");
  const output = document.getElementById("json-output");
  message.textContent = "正在分析照片與記憶資料庫...";
  output.textContent = "{}";

  const passageImages = {};
  for (const [label, id] of [
    ["left", "passage-left"],
    ["front", "passage-front"],
    ["right", "passage-right"],
    ["back", "passage-back"],
  ]) {
    const record = await optionalPassageRecord(id);
    if (record) {
      passageImages[label] = record;
    }
  }

  const payload = {
    target_room_id: document.getElementById("target-room").value.trim(),
    waypoint_room_ids: document.getElementById("waypoints").value.trim(),
    localization_images: localizationRecords,
    passage_images: passageImages,
  };

  try {
    const response = await fetch("/api/guide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    message.textContent = result.message_zh || result.rationale_zh || "系統已回傳結果。";
    output.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    message.textContent = `送出失敗：${error}`;
  }
});

renderLocalizationMemory();
