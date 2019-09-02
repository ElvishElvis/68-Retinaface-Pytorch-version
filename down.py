from requests import get  # to make GET request


def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        print("runinng!!!!!")
        # get request
        response = get(url)
        # write to file
        print("get {}".format(file_name))

        file.write(response.content)

download("https://www.adrianbulat.com/downloads/FaceAlignment/LS3D-W-balanced-20-03-2017.zip ",'sample.zip')
download('https://uniofnottm-my.sharepoint.com/personal/adrian_bulat_nottingham_ac_uk/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fadrian%5Fbulat%5Fnottingham%5Fac%5Fuk%2FDocuments%2FUoN%20Box%20Migration%2FPublic%2FLS3D%2DW%2FLS3D%2DW%2Etar%2Egz','all.zip')


