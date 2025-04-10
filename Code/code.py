# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.common.exceptions import NoSuchElementException
# import time

# driver = webdriver.Chrome()

# driver.get("https://www.coursera.org/search?query=")

# courses_collected = 0
# target_courses = 10000  
# scroll_pause_time = 0.8  
# scroll_increment = 1500 

# seen_courses = set()
# course_data = []

# while courses_collected < target_courses:
#     cards = driver.find_elements(By.CSS_SELECTOR, "div.cds-ProductCard-gridCard")

#     for card in cards:
#         try:
#             title = card.find_element(By.CSS_SELECTOR, "h3.cds-CommonCard-title").text.strip()
#         except NoSuchElementException:
#             title = ""

#         if title and title in seen_courses:
#             continue

#         try:
#             partner = card.find_element(By.CSS_SELECTOR, "div.cds-ProductCard-partners p.cds-ProductCard-partnerNames").text.strip()
#         except NoSuchElementException:
#             partner = ""

#         try:
#             rating = card.find_element(By.CSS_SELECTOR, "div.cds-RatingStat-sizeLabel span.css-6ecy9b").text.strip()
#         except NoSuchElementException:
#             rating = ""

#         try:
#             reviews = card.find_element(By.CSS_SELECTOR, "div.cds-RatingStat-sizeLabel div.css-vac8rf").text.strip()
#         except NoSuchElementException:
#             reviews = ""

#         try:
#             description = card.find_element(By.CSS_SELECTOR, "div.cds-CommonCard-metadata p.css-vac8rf").text.strip()
#         except NoSuchElementException:
#             description = ""

#         try:
#             course_link = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
#         except NoSuchElementException:
#             course_link = ""

#         if title:
#             seen_courses.add(title)
#             course_data.append({
#                 "title": title,
#                 "partner": partner,
#                 "rating": rating,
#                 "reviews": reviews,
#                 "description": description,
#                 "course_link": course_link
#             })
#             courses_collected += 1

#             print(f"Scraped {courses_collected}/{target_courses} courses.", end="\r")

#             if courses_collected >= target_courses:
#                 break

#     driver.execute_script(f"window.scrollBy(0, {scroll_increment});")
#     time.sleep(scroll_pause_time)

# driver.quit()

# df = pd.DataFrame(course_data)

# df.to_csv("coursera_courses.csv", index=False)

# print(f"\nCollected {courses_collected} courses.")
# for idx, course in enumerate(course_data, start=1):
#     print(f"Course {idx}: {course}")
