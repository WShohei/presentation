#import "templates/template.typ": *
#show: doc => slides(doc)

#show link: l => underline(l)

#slide(title: "Motivation")[
- The template aims to make it easy to create pretty slides
- Typst is great to use
- Automatic styling of sections/chapters via headings
- `#slide` function for creating slides
]

= How to use

#slide(
  title: "Starting with the template", pad(
    x: 2em,
  )[
  + このスライドはpytorchのチュートリアルを作成するためのテンプレートです。
  + Adjust the metadata in the `metadata.typ` file to your needs.
  + Create a `slides.typ` or similar with the following starting content:

  ```typst
                      #import "template.typ": *
                      #show: doc => slides(doc)
                      ```
  ],
)

#slide(
  title: "Filling in content", pad(
    x: 2em,
  )[
  Next you can fill in content with headings for sections and `slide` functions
  for slides:

  ```typst
                      = My Section

                      #slide(title: "My Title")[
                        My slide content.
                      ]

                      #slide()[
                        #image("your-image.svg", width: 80%)
                      ]
                      ```
  ],
)

#slide(title: "Drawings")[
  #circle(radius: 5em, fill: red)
]

= Problems

#slide(
  pad(
    x: 2em,
  )[
  - There is no thing similar to `\pause` sadly, at least I have not found a way to
    do it
  ],
)
