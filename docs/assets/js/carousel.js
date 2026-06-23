
(function() {
  function initCarousel() {
    const carousel = document.querySelector('.report-carousel');
    if (!carousel) {
      return;
    }

    const slides = carousel.querySelectorAll('.carousel-slide');
    const totalSlides = slides.length;
    if (totalSlides === 0) {
      return;
    }

    let currentSlide = 0;

    const prevBtn = carousel.querySelector('#carousel-prev');
    const nextBtn = carousel.querySelector('#carousel-next');
    const pageInfo = carousel.querySelector('#carousel-page');
    const dotsContainer = carousel.querySelector('#carousel-dots');

    if (!prevBtn || !nextBtn || !pageInfo || !dotsContainer) {
      return;
    }

    // Create dots
    dotsContainer.innerHTML = '';
    for (let i = 0; i < totalSlides; i++) {
      const dot = document.createElement('div');
      dot.classList.add('carousel-dot');
      if (i === 0) dot.classList.add('active');
      dot.setAttribute('data-slide', i);
      dot.addEventListener('click', function() {
        goToSlide(i);
      });
      dotsContainer.appendChild(dot);
    }

    const dots = carousel.querySelectorAll('.carousel-dot');

    function updateCarousel() {
      slides.forEach((slide, i) => {
        if (i === currentSlide) {
          slide.classList.add('active');
        } else {
          slide.classList.remove('active');
        }
      });

      dots.forEach((dot, i) => {
        if (i === currentSlide) {
          dot.classList.add('active');
        } else {
          dot.classList.remove('active');
        }
      });

      pageInfo.textContent = 'Page ' + (currentSlide + 1) + ' of ' + totalSlides;
      prevBtn.disabled = currentSlide === 0;
      nextBtn.disabled = currentSlide === totalSlides - 1;
    }

    function goToSlide(n) {
      currentSlide = Math.max(0, Math.min(n, totalSlides - 1));
      updateCarousel();
    }

    prevBtn.addEventListener('click', function(e) {
      e.preventDefault();
      if (currentSlide > 0) {
        goToSlide(currentSlide - 1);
      }
    });

    nextBtn.addEventListener('click', function(e) {
      e.preventDefault();
      if (currentSlide < totalSlides - 1) {
        goToSlide(currentSlide + 1);
      }
    });

    document.addEventListener('keydown', function(e) {
      if (e.key === 'ArrowLeft' && currentSlide > 0) goToSlide(currentSlide - 1);
      if (e.key === 'ArrowRight' && currentSlide < totalSlides - 1) goToSlide(currentSlide + 1);
    });

    updateCarousel();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCarousel);
  } else {
    initCarousel();
  }
})();
