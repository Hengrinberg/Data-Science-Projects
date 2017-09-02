from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib2
import xlsxwriter
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


########## web scraping #####################
# An automated program that goes to wikipedia site find the languages ,external links and pictures
# for different values and write the result for each value in a separate excel spreadsheet.

def wiki_data_import(keyword, workbook, driver):
    first_result.clear()
    first_result.send_keys(keyword)
    search_button.click()

    counter = 0
    while counter <= 3:
        try:
            # elems = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//a[@class='interlanguage-link-target']")))
            # links = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//a[@href]")))
            # picks = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//img[@class='thumbimage']")))
            WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[@class='interlanguage-link-target']")))
            WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//a[@href]")))
            WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.XPATH, "//img[@class='thumbimage']")))
            break
        except TimeoutException as ex:
            print("Exception has been thrown. " + str(ex))
            counter += 1

    # find languages
    elems = driver.find_elements_by_xpath("//a[@class='interlanguage-link-target']")
    languages = []
    for element in elems:
        languages.append(element.text)

    # find external links
    links = driver.find_elements_by_xpath("//a[@href]")
    elinks = []
    for link in links:
        elinks.append(link.get_attribute("href"))

    # find pictures
    picslist = []
    pics = driver.find_elements_by_xpath("//img[@class='thumbimage']")
    for pic in pics:
        picslist.append(pic.get_attribute("src"))

    num_of_pics = len(picslist)

    data = {"external links": elinks, "languages": languages, "pictures": num_of_pics}
    data["pictures"] = [num_of_pics]

    ################creation of a new excel file #######################

    worksheet = workbook.add_worksheet()
    raw = 0
    col = 0
    worksheet.write(raw, col, "external links")
    worksheet.write(raw, col + 1, "languages")
    worksheet.write(raw, col + 2, "pictures")
    raw = 1
    col = 0
    para1 = data["external links"]
    for value in para1:
        worksheet.write(raw, col, value)
        raw += 1

    raw = 1
    col = 1
    para2 = data["languages"]
    for value in para2:
        worksheet.write(raw, col, value)
        raw += 1

    raw = 1
    col = 2
    para2 = data["pictures"]
    for value in para2:
        worksheet.write(raw, col, value)
        raw += 1

    driver.back()


######## test #######
driver = webdriver.Firefox()
driver.get('https://en.wikipedia.org/wiki/')

input_ = ['infection', 'adar', 'moon', 'world', 'max', 'beautiful', 'world', 'hello world']
workbook = xlsxwriter.Workbook('guiautomation2.xlsx')
for keyword in input_:
    counter = 0
    while counter <= 6:
        try:
            first_result = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchInput")))
            search_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchButton")))
            break

        except TimeoutException as ex:
            print("Exception has been thrown. " + str(ex))
            counter += 1

    wiki_data_import(keyword, workbook, driver)

workbook.close()
driver.quit()