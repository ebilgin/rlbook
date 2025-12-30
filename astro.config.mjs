import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import cloudflare from '@astrojs/cloudflare';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://rlbook.ai',

  integrations: [
    mdx(),
    react(),
    tailwind(),
  ],

  // Cloudflare Pages adapter for edge deployment
  output: 'static',
  adapter: cloudflare(),

  // Markdown/MDX configuration
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },

  // Vite configuration for client-side bundling
  vite: {
    optimizeDeps: {
      exclude: ['@tensorflow/tfjs'],
    },
    ssr: {
      noExternal: ['katex'],
    },
  },
});
