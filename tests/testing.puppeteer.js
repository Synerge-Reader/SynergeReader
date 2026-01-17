import puppeteer from 'puppeteer';
import { generalQuestions } from './testing.data.js';

const generalResults = []
const medicalResults = []
const lawResults = []

const browser = await puppeteer.launch({
    headless: false, // if you want to see the full exec change to true
    defaultViewport: { width: 768, height: 768 }
})

const testResults = await Promise.allSettled([
    generalModelEvaluation(),
    medicalModelEvaluation(),
    lawModelEvaluation()
])

console.log("done")
//await browser.close()




//console.log(generalQuestions[0])






async function generalModelEvaluation() {
    try {
        /*

               let generalPage = await browser.newPage()
               generalPage.setDefaultNavigationTimeout(3 * 60 * 1000) // feel free to tweak this value this is just because sometimes are responses are slow
               await generalPage.goto("http://localhost:3000", { waitUntil: "networkidle0" })
       
       
               const fileElement = await generalPage.waitForSelector('input[type=file]');
               await fileElement.uploadFile('./tests/Documents/photosynthesis_overview.pdf');
       
       
       
       
               const textArea = await generalPage.locator('textarea#question')
               // await textArea.fill(userMessage)
       
       
       
       
       
       
       
       
       
       
              
               const uploadFile = await generalPage.locator('button.alpha-upload-btn').click()
               await uploadFile.SetInputFiles("./tests/Documents/photosynthesis_overview.pdf")
               //const textField = await generalPage.$('.modal-content form')
               //const textField = await generalPage.locator('div.main-action-box')
                */

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


        console.log("CLICKED")

        let userMessage = "yap yap yap"

        const textArea = await medicalPage.locator('textarea#question')
        await textArea.fill(userMessage)


        console.log("Do Something medical")

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

        let userMessage = "yap yap yap"

        const textArea = await medicalPage.locator('textarea#question')
        await textArea.fill(userMessage)


    }
    catch (err) {

        console.log(err.message)


    }


}

