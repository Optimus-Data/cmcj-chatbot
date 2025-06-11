require("dotenv").config();
const fs = require("fs");
const path = require("node:path");
const { Document } = require("@langchain/core/documents");
const DATABASE_DIRECTORY = path.join(process.cwd(), "database");
const DOCS_PATH = path.join(DATABASE_DIRECTORY, "docs.json");

async function loadDocumentsFromJson(filePath) {
    console.log(`Processando arquivo JSON: ${filePath}`);
    try {
        const fileContent = fs.readFileSync(filePath, "utf-8");
        const jsonData = JSON.parse(fileContent);
        const documents = [];
        const fileName = path.basename(filePath);

        if (fileName === "data.json") {
            for (let i = 0; i < jsonData.length; i++) {
                const item = jsonData[i];

                if (
                    item.introduction &&
                    item.id &&
                    item.body &&
                    item.title &&
                    item.bible_reference &&
                    item.author
                ) {
                    const pageContent =
                        `Título: ${item.title}; Autor: ${item.author} ; Data ${ item.data }\n` +
                        `Referência Bíblica: ${item.bible_reference}\n` +
                        `Introdução: ${item.introduction}\n` +
                        `Corpo: ${item.body}`
                    const doc = new Document({
                        pageContent: pageContent,
                        metadata: {
                            source: fileName,
                            type: "data",
                            id: item.id,
                            author: item.author,
                            index: i,
                        },
                    });
                    documents.push(doc);
                } else {
                    console.log(`Item inválido no índice ${i}`);
                }
            }
        }

        console.log(
            `Arquivo ${fileName} processado. ${documents.length} documentos criados.`,
        );
        return documents;
    } catch (error) {
        console.error(`Erro ao processar o arquivo JSON ${filePath}:`, error);
        return [];
    }
}

module.exports = {
    loadDocumentsFromJson,
};
