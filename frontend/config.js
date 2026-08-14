/* Split-deploy configuration.
 *
 * Leave this file out (or leave the value empty) when the FastAPI service
 * serves the frontend itself — app.js then calls its own origin, which is
 * the default single-service setup.
 *
 * Only set NEXATEL_API when the frontend is hosted separately, e.g. static
 * on Vercel with the API on Render. Include this file before app.js in
 * index.html when you do.
 */
window.NEXATEL_API = "https://nexatel-churn-intelligence.onrender.com";
