// Initialisation carte Leaflet + outils de dessin AOI
const map = L.map('map').setView([34.0, -6.8], 6); // Zoom sur le Maroc par défaut

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const drawControl = new L.Control.Draw({
  edit: { featureGroup: drawnItems },
  draw: {
    polygon: true,
    rectangle: true,
    polyline: false,
    circle: false,
    marker: false,
  }
});
map.addControl(drawControl);

let currentAOI = null;

map.on(L.Draw.Event.CREATED, function (e) {
  const layer = e.layer;
  drawnItems.clearLayers();
  drawnItems.addLayer(layer);
  currentAOI = layer.toGeoJSON();
  L.popup()
    .setLatLng(layer.getBounds ? layer.getBounds().getCenter() : layer.getLatLng())
    .setContent('AOI sélectionnée. Prête pour le calcul.')
    .openOn(map);
});

document.getElementById('google-login').addEventListener('click', () => {
  alert('Login Google: placeholder. À intégrer via Google Identity.');
});

document.getElementById('compute-ndvi').addEventListener('click', async () => {
  if (!currentAOI) {
    alert('Veuillez dessiner une AOI (polygone/rectangle) sur la carte.');
    return;
  }
  // Placeholder: appeler l'API backend /gee/indices avec NDVI
  const params = new URLSearchParams({ index: 'NDVI', collection: 'SENTINEL_2' });
  try {
    const res = await fetch(`http://localhost:8000/gee/indices?${params.toString()}`);
    const data = await res.json();
    alert('Réponse backend (placeholder): ' + JSON.stringify(data));
  } catch (err) {
    console.error(err);
    alert('Erreur lors de l’appel API (assurez-vous que le backend est lancé).');
  }
});