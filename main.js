import { pipeline } from '@huggingface/transformers';
const segmenter = await pipeline("background-removal", "Xenova/modnet", { device: "cpu", dtype: "fp32" });
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/woman-with-afro_medium.jpg';
const output = await segmenter(url);
output[0].save("output.png"); // (Optional) Save the image