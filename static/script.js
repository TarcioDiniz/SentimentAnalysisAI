document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("sentiment-form");
  const textoField = document.getElementById("texto");
  const resultadoDiv = document.getElementById("resultado");
  const erroDiv = document.getElementById("erro");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultadoDiv.innerHTML = "";
    erroDiv.innerHTML = "";

    const texto = textoField.value.trim();
    if (!texto) {
      erroDiv.innerText = "Por favor, insere um texto antes de analisar.";
      return;
    }

    try {
      const response = await fetch("/analyze-sentiment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: texto })
      });

      if (!response.ok) {
        const err = await response.json();
        erroDiv.innerText = "Erro: " + (err.detail || "Algo correu mal.");
        return;
      }

      const data = await response.json();
      resultadoDiv.innerHTML = `
        <h3>Resultado:</h3>
        <ul>
          <li><strong>neg:</strong> ${data.neg}</li>
          <li><strong>neu:</strong> ${data.neu}</li>
          <li><strong>pos:</strong> ${data.pos}</li>
          <li><strong>compound:</strong> ${data.compound}</li>
        </ul>
      `;
    } catch (error) {
      erroDiv.innerText = "Erro de rede: não foi possível contactar o servidor.";
      console.error(error);
    }
  });
});
