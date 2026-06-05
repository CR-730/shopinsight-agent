import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Sohne"',
          '"Inter"',
          '"Noto Sans SC"',
          '"PingFang SC"',
          '"Microsoft YaHei"',
          "sans-serif",
        ],
        mono: ['"SFMono-Regular"', "Consolas", '"Liberation Mono"', "monospace"],
      },
      colors: {
        chatgpt: {
          green: "#10a37f",
          text: "#202123",
          muted: "#565869",
          subtle: "#8e8ea0",
          sidebar: "#f9f9f9",
          surface: "#ffffff",
          bubble: "#f4f4f4",
          border: "#ececec",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
