from xml.dom.minidom import parse, Text
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.alert import Alert


############## xml exercise ##################
# An automated program that fill all the checkboxes in the the web page that I created before and
# then take the confirmation (hashed value) and write it back to the xml file


def agree_terms(driver):
    ''' a function that fill all the checkboxes '''
    empty_tabs = driver.find_element_by_xpath("//input[@id='1']")
    empty_tabs.click()
    read = driver.find_element_by_xpath("//input[@id='2']")
    read.click()
    agree = driver.find_element_by_xpath("//input[@id='3']")
    agree.click()


driver = webdriver.Firefox()
url = "file:///home/genesis/git/hen-grinberg/html/hash.html"
driver.get(url)

doc = parse("guiauto2.xml")
persons = doc.getElementsByTagName("person")

for person in persons:
    first_ = driver.find_element_by_xpath("//input[@id='fname']")
    last_ = driver.find_element_by_xpath("//input[@id='Last name']")
    year_ = driver.find_element_by_xpath("//select[@id='year']")
    month_ = driver.find_element_by_xpath("//select[@id='month']")
    day_ = driver.find_element_by_xpath("//select[@id='day']")
    male_ = driver.find_element_by_xpath("//input[@id='male']")
    female_ = driver.find_element_by_xpath("//input[@id='female']")
    submit_ = driver.find_element_by_xpath("//button[@id='submit']")
    firstname = person.getElementsByTagName("firstName")[0].firstChild.data
    first_.clear()
    first_.send_keys(firstname)
    lastname = person.getElementsByTagName("lastName")[0].firstChild.data
    last_.clear()
    last_.send_keys(lastname)
    gender = person.getElementsByTagName("gender")[0].firstChild.data
    if gender == "Male":
        male_.click()
    else:
        female_.click()
    b_date = person.getElementsByTagName("birthDay")[0].firstChild.data
    birthdate_d = b_date.split("/")[0]
    for option in day_.find_elements_by_tag_name("option"):
        if option.text == birthdate_d:
            option.click()
            break
    birthdate_m = b_date.split("/")[1]
    for option in month_.find_elements_by_tag_name("option"):
        if option.text == birthdate_m:
            option.click()
            break
    birthdate_y = b_date.split("/")[2]
    for option in year_.find_elements_by_tag_name("option"):
        if option.text == birthdate_y:
            option.click()
            break
    agree_terms(driver)
    submit_.click()

    try:
        WebDriverWait(driver, 10).until(EC.alert_is_present())
        alert = driver.switch_to_alert()
        hash_ = alert.text  # handle the content of the alert
        alert.accept()  # accepts the alert note


    except TimeoutException:
        print " alert window expected "

    driver.get(url)  # open new tab

    hash_tag = person.getElementsByTagName("hash")[0]  # finds the hash tag
    hash_value = Text()  # new object that will handle the alert content
    hash_value.data = hash_  # assign the alert content into the text object
    hash_tag.appendChild(hash_value)  # append the object into the dom

xml = open("guiautotest.xml", "w")
doc.writexml(xml)  # write the new dom to a new xml file
driver.quit()
