import puppeteer from 'puppeteer';
import { generalQuestions, medicalQuestions, lawQuestions } from './testing.data.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const generalResults = []
const medicalResults = []
const lawResults = []

const browser = await puppeteer.launch({
    headless: false, // if you want to see the full exec change to true
    defaultViewport: { width: 768, height: 768 },
    args: ['--start-maximized']
})



const testResults = await Promise.allSettled([
    generalModelEvaluation(),
    medicalModelEvaluation(),
    lawModelEvaluation()
])

// write to csv later when stable
console.log(medicalResults)
console.log(lawResults)
console.log(generalResults)
console.log("done")
await browser.close()



async function generalModelEvaluation() {

    try {


        let generalPage = await browser.newPage()
        generalPage.setDefaultNavigationTimeout(3 * 60 * 1000) // feel free to tweak this value this is just because sometimes are responses are slow
        await generalPage.goto("http://localhost:3000", { waitUntil: "networkidle0" })

        const fileInput = await generalPage.waitForSelector('input[type=file]');
        await fileInput.uploadFile('./tests/Documents/photosynthesis_overview.pdf');


        const qOptions = await generalPage.$$('.main-text-Box button')
        const askQuestion = qOptions[1]




        const textArea = await generalPage.locator('textarea#question')


        for (const gQuestion of generalQuestions) {

            await textArea.fill(gQuestion.q)
            await askQuestion.click()
            await new Promise(resolve => setTimeout(resolve, 2 * 60 * 1000));
            // await generalPage.wait(2 * 60 * 1000)

            const modelResponseText = await generalPage.$eval(
                '.main-action-box p',
                el => el.textContent
            );


            const result = {
                question: gQuestion.q,
                expectedAnswer: gQuestion.expectedAnswer,
                modelAnswer: modelResponseText
            }

            generalResults.push(result)

        }

        //const textField = await generalPage.$('.modal-content form')
        //const textField = await generalPage.locator('div.main-action-box')


    }
    catch (err) {
        console.log(err.message)
    }

}

async function medicalModelEvaluation() {
    try {


        let medicalPage = await browser.newPage()

        medicalPage.setDefaultNavigationTimeout(3 * 60 * 1000)
        await medicalPage.goto("http://localhost:3000", { waitUntil: ["networkidle0", "domcontentloaded"] })
        const clickDropDown = await medicalPage.$('.domain-dropdown h3')
        await clickDropDown.click()

        const domains = await medicalPage.$$('.dropdown-menu div')
        const medicalDomain = domains[0]
        await medicalDomain.click()

        const fileInput = await medicalPage.waitForSelector('input[type=file]');
        await fileInput.uploadFile('./tests/Documents/hypertension_paper.pdf');


        const qOptions = await medicalPage.$$('.main-text-Box button')
        const askQuestion = qOptions[1]




        const textArea = await medicalPage.locator('textarea#question')

        for (const mQuestion of medicalQuestions) {

            await textArea.fill(mQuestion.q)
            await askQuestion.click()
            //  await medicalPage.wait(3 * 60 * 1000)
            await new Promise(resolve => setTimeout(resolve, 2 * 60 * 1000));

            const modelResponseText = await medicalPage.$eval(
                '.main-action-box p',
                el => el.textContent
            );


            const result = {
                question: mQuestion.q,
                expectedAnswer: mQuestion.expectedAnswer,
                modelAnswer: modelResponseText
            }

            medicalResults.push(result)

        }

    }
    catch (err) {

        console.log(err.message)

    }
}

async function lawModelEvaluation() {

    try {


        let lawPage = await browser.newPage()
        lawPage.setDefaultNavigationTimeout(3 * 60 * 1000)
        await lawPage.goto("http://localhost:3000", { waitUntil: ["networkidle0", "domcontentloaded"] })
        const clickDropDown = await lawPage.$('.domain-dropdown h3')
        await clickDropDown.click()

        const domains = await lawPage.$$('.dropdown-menu div')
        const lawDomain = domains[1]
        await lawDomain.click()

        const fileInput = await lawPage.waitForSelector('input[type=file]');
        await fileInput.uploadFile('./tests/Documents/constitutional_law_paper.pdf');

        const qOptions = await lawPage.$$('.main-text-Box button')
        const askQuestion = qOptions[1]

        const textArea = await lawPage.locator('textarea#question')

        for (const lQuestion of lawQuestions) {

            await textArea.fill(lQuestion.q)
            await askQuestion.click()
            //  await lawPage.wait(3 * 60 * 1000)
            await new Promise(resolve => setTimeout(resolve, 2 * 60 * 1000));

            const modelResponseText = await lawPage.$eval(
                '.main-action-box p',
                el => el.textContent
            );

            const result = {
                question: lQuestion.q,
                expectedAnswer: lQuestion.expectedAnswer,
                modelAnswer: modelResponseText
            }

            lawResults.push(result)

        }


    }
    catch (err) {

        console.log(err.message)


    }


}

