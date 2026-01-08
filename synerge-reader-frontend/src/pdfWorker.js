import { GlobalWorkerOptions } from 'pdfjs-dist/build/pdf';

const pdfWorkerUrl = new URL('pdfjs-dist/build/pdf.worker.min.js', import.meta.url);
GlobalWorkerOptions.workerSrc = pdfWorkerUrl.href;
