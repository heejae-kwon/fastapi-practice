## Remove Background automatically with 1 API call

Explore our API documentation and examples to integrate remove.bg into your application or workflow.

**[Get API Key](https://www.remove.bg/dashboard#api-key)**

**Not a developer?**There are plenty ready to use [Tools & Apps](https://www.remove.bg/tools-api) by remove.bg and our community.

![https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/removebg-in-and-out-fe6e8a0859d78885b7f60f125c6dbb5cd97d54474a4132f8173f8d434fe01e46.gif](https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/removebg-in-and-out-fe6e8a0859d78885b7f60f125c6dbb5cd97d54474a4132f8173f8d434fe01e46.gif)

## Easy to integrate

Our API is a simple HTTP interface with various options:

- **Source images:** Direct uploads or URL reference
- **Result images:** Image file or JSON-encoded data
- **Output resolution:** up to 25 megapixels

Requires images that [have a foreground (e.g. people, products, animals, cars, etc.)](https://www.remove.bg/help/a/what-images-are-supported)

![https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api_thumb-44ec09edd3277ac04003a42f57bd7af5b1aff2cd1443a115878028c38df64a65.jpg](https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api_thumb-44ec09edd3277ac04003a42f57bd7af5b1aff2cd1443a115878028c38df64a65.jpg)

## Get started

Our API is a simple HTTP interface with various options:

1. [Get your API Key](https://www.remove.bg/dashboard#api-key).Your first 50 API calls per month are on us (see [Pricing](https://www.remove.bg/pricing)).
2. Use the following code samples to get started quickly
3. Review the reference docs to adjust any parameters

![https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/get-started-3d9f59c339263694315c80489ce941039f1c10b8471cee577caac54d4b688da5.png](https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/get-started-3d9f59c339263694315c80489ce941039f1c10b8471cee577caac54d4b688da5.png)

## Sample Code

**Image File**

```bash
$ curl -H 'X-API-Key: INSERT_YOUR_API_KEY_HERE'           \
       -F 'image_file=@/path/to/file.jpg'                 \
       -F 'size=auto'                                     \
       -f https://api.remove.bg/v1.0/removebg -o no-bg.png
```

## Output formats

You can request one of three formats via the `format` parameter:

| Format | Resolution | Pros and cons | Example |
| --- | --- | --- | --- |
| PNG | Up to 10 Megapixelse.g. 4000x2500 | + Simple integration+ Supports transparency- Large file size | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.pnghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.pnghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.png |
| JPG | Up to 25 Megapixelse.g. 6250x4000 | + Simple Integration+ Small file size- No transparency supported | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpg |
| ZIP | Up to 25 Megapixelse.g. 6250x4000 | + Small file size+ Supports transparency- Integration requires compositing | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.ziphttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.ziphttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.zip |

Please note that **PNG images above 10 megapixels are not supported**. If you require transparency for images of that size, use the ZIP format (see below). If you don't need transparency (e.g. white background), we recommend JPG.

## How to use the ZIP format

The ZIP format has the best runtime performance for transparent images.

In comparison to PNG, the resulting file is up to 80% smaller (faster to download) and up to 40% faster to generate. For performance optimization we recommend using the ZIP format whenever possible. Above 10 megapixels, usage of the ZIP format is required for transparent results.

The ZIP file always contains the following files:

| color.jpg | A non-transparent RGB image in JPG format containing the colors for each pixel. (Note: This image differs from the input image due to edge color corrections.) – https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-color-a80a31a9105024481c0629bac2c5a2121d54ff09923f68ac4f5b0a3a291346d3.jpg |
| --- | --- |
| alpha.png | A non-transparent gray scale image in PNG format containing the alpha matte. White pixels are foreground regions, black is background. – https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-alpha-e0ae0f94e34a25f1e9bd7a2a68a0bb34f1568a5bcdadd62e623b5cb021981032.png |

To compose the final image file:

1. Extract the files from the ZIP
2. Apply the alpha matte (alpha.png) to the color image (color.jpg).
3. Save the result in a format of your choice (e.g. PNG)

**Sample code for Linux** (requires [zip](http://infozip.sourceforge.net/) and [imagemagick](https://imagemagick.org/)):

```bash
$ unzip no-bg.zip                      && \
  convert color.jpg alpha.png             \
    -compose CopyOpacity                  \
    -composite no-bg.png

```

A `zip2png` command is integrated in our [command line interface](https://github.com/remove-bg/remove-bg-cli/). More code samples can be found [here](https://github.com/remove-bg/integration/tree/master/).

## Rate Limit

You can process up to **500 images per minute** through the API, depending on the input image resolution in megapixels.

Examples:

| Input image | Megapixels | Effective Rate Limit |
| --- | --- | --- |
| 625 x 400 | 1 MP | 500 images per minute |
| 1200 x 800 | 1 MP | 500 images per minute |
| 1600 x 1200 | 2 MP | 500 / 2 = 250 images per minute |
| 2500 x 1600 | 4 MP | 500 / 4 = 125 images per minute |
| 4000 x 2500 | 10 MP | 500 / 10 = 50 images per minute |
| 6250 x 4000 | 25 MP | 500 / 25 = 20 images per minute |

Exceed of rate limits leads to a HTTP status 429 response (no credits charged). Clients can use the following response headers to gracefully handle rate limits:

| Response header | Sample value | Description |
| --- | --- | --- |
| X-RateLimit-Limit | 500 | Total rate limit in megapixel images |
| X-RateLimit-Remaining | 499 | Remaining rate limit for this minute |
| X-RateLimit-Reset | 1696915907 | Unix timestamp when rate limit will reset |
| Retry-After | 59 | Seconds until rate limit will reset (only present if rate limit exceeded) |

Higher Rate Limits are available [upon request](https://www.remove.bg/support/contact?subject=Rate+Limit+Requirements).

## Exponential back-off

Exponential back-off is an error handling strategy in which a client periodically retries a failed request.

The delay increases between requests and often includes jitters (randomized delay) to avoid collisions when using concurrent clients. Clients should use exponential back-off whenever they receive HTTP status codes `5XX` or `429`.

The following pseudo code shows one way to implement exponential back-off:

```php
retry = true
retries = 0

WHILE (retry == TRUE) AND (retries < MAX_RETRIES)
  wait_for_seconds (2^retries + random_number)

  result = get_result

  IF result == SUCCESS
    retry = FALSE
  ELSE IF result == ERROR
    retry = TRUE
  ELSE IF result == THROTTLED
    retry = TRUE
  END

  retries = retries +1
END

```

## API Change-log

Most recent API updates:

- **2021-12-07:** Added foreground position and size to background removal responses. (JSON fields: `foreground_top`, `foreground_left`, `foreground_width` and `foreground_height`. Response headers: `X-Foreground-Top`, `X-Foreground-Left`, `X-Foreground-Width` and `X-Foreground-Height`.)
- **2021-04-13:** Removed deprecated `shadow_method=legacy` option and `shadow_method` parameter as it no longer has any effect.
- **2021-03-01:** Added examples for `400` error codes.
- **2021-01-21:** Added `shadow_method` parameter to control shadow appearance. Deprecated `legacy` value for `shadow_method`.
- **2020-09-30:** Added `type_level` parameter and `POST /improve` endpoint.
- **2020-05-06:** Added `semitransparency` parameter.
- **2019-09-27:** Introduce ZIP format and support for images up to 25 megapixels.
- **2019-09-25:** Increased maximum file size to 12 MB and rate limit to 500 images per minute. Rate limit is now resolution-dependent.
- **2019-09-16:** Added `enterprise` credit balance field to account endpoint.
- **2019-08-01:** Added `add_shadow` option for car images.
- **2019-06-26:** Added `crop_margin`, `scale` and `position` parameters.
- **2019-06-19:** Added support for animals and other foregrounds (response header `X-Type: animal` and `X-Type: other`)
- **2019-06-11:** Credit balances can now have fractional digits, this affects the `X-Credits-Charged` value
- **2019-06-03:** Added parameters `bg_image_url`, `bg_image_file` (add a background image), `crop` (crop off empty regions) and `roi` (specify a region of interest).
- **2019-05-13:** Added car support (`type=car` parameter and `X-Type: car` response header)
- **2019-05-02:** Renamed size `"regular"` to `"small"`. Clients should use the new value, but the old one will continue to work (deprecated)
- **2019-05-01:** Added endpoint `GET /account` for credit balance lookup
- **2019-04-15:** Added parameter `format` to set the result image format
- **2019-04-15:** Added parameter `bg_color` to add a background color to the result image
