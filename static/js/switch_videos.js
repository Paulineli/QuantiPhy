function switchVideo(prefix, videoContainerId, preview_id) {
    const totalVideos = 10;

    // Hide all video containers and pause/reset their videos
    for (let i = 1; i <= totalVideos; i++) {
        const container = document.getElementById(`${prefix}video${i}Container`);
        if (container) {
            container.style.display = 'none'; // Hide the video container
            
            // Pause all videos and reset to start
            const videos = container.getElementsByTagName('video');
            Array.from(videos).forEach(video => {
                video.pause();
                video.currentTime = 0;
            });
        }
    }

    // Show the selected video container
    const selectedContainer = document.getElementById(`${prefix}${videoContainerId}`);
    if (selectedContainer) {
        selectedContainer.style.display = 'block'; // Show the selected video
        const selectedVideo = selectedContainer.querySelector('video');
        if (selectedVideo) {
            selectedVideo.play(); // Autoplay the selected video
        }
    }

    // Update preview images: remove active class from all previews
    for (let i = 1; i <= totalVideos; i++) {
        const preview = document.getElementById(`${prefix}video${i}Preview`);
        if (preview) {
            preview.classList.remove("preview-video-active");
        }
    }

    // Add active class to the selected preview
    const selectedPreview = document.getElementById(`${prefix}${preview_id}`);
    if (selectedPreview) {
        selectedPreview.classList.add("preview-video-active");
    }
}

function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function checkVideoVisibility() {
    const videos = document.querySelectorAll('.auto-video');
    videos.forEach(video => {
        if (isElementInViewport(video)) {
            if (video.paused) {
                video.currentTime = 0; // Reset to start
                video.play();
            }
        } else {
            video.pause();
        }
    });
}

window.addEventListener('scroll', checkVideoVisibility);

document.addEventListener("DOMContentLoaded", function() {
    // Set default video volume for all '.video-music' videos
    const videos = document.querySelectorAll('.video-music');
    videos.forEach(video => {
        video.volume = 0.25; // Set volume to 25%
    });

    // Set video 2 as active on load (since video 1 was removed)
    const firstVideoContainer = document.getElementById("Touristvideo2Container");
    if (firstVideoContainer) {
        firstVideoContainer.style.display = "block"; // Display the first video
        const firstVideo = firstVideoContainer.querySelector('video');
        if (firstVideo) {
            firstVideo.play();
        }
    }

    const firstPreview = document.getElementById("Touristvideo2Preview");
    if (firstPreview) {
        firstPreview.classList.add("preview-video-active"); // Highlight first preview
    }
});
