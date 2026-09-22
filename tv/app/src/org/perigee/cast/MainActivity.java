package org.perigee.cast;

import android.app.Activity;
import android.net.http.SslCertificate;
import android.net.http.SslError;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.webkit.SslErrorHandler;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import java.security.MessageDigest;
import java.security.cert.X509Certificate;

/** Full-screen, JavaScript-free viewer of the Perigee cast stream. Talks to one host, over TLS,
 *  only if the server presents the exact pinned certificate, and only with the built-in token. */
public class MainActivity extends Activity {

    private WebView web;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(false);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setDomStorageEnabled(false);
        s.setSaveFormData(false);
        s.setCacheMode(WebSettings.LOAD_NO_CACHE);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        web.setBackgroundColor(0xFF000000);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                // Never navigate anywhere but the cast host.
                return !CastConfig.HOST.equals(req.getUrl().getHost());
            }

            @Override
            public void onReceivedSslError(WebView v, SslErrorHandler handler, SslError error) {
                // Self-signed server: accept only the one certificate whose SHA-256 is baked in.
                if (fingerprintMatches(error.getCertificate())) handler.proceed(); else handler.cancel();
            }

            @Override
            public void onPageFinished(WebView v, String url) { hideSystemUi(); }
        });

        setContentView(web);
        hideSystemUi();
        web.loadUrl(CastConfig.URL);
    }

    private static boolean fingerprintMatches(SslCertificate cert) {
        try {
            X509Certificate x = SslCertificate.saveState(cert) == null ? null : certFrom(cert);
            if (x == null) return false;
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(x.getEncoded());
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format("%02x", b));
            return hex.toString().equalsIgnoreCase(CastConfig.CERT_SHA256);
        } catch (Exception e) {
            return false;
        }
    }

    private static X509Certificate certFrom(SslCertificate cert) {
        try {
            // API 29+: getX509Certificate(); older Fire OS: pull it out of the saved state bundle
            java.lang.reflect.Method m = SslCertificate.class.getMethod("getX509Certificate");
            return (X509Certificate) m.invoke(cert);
        } catch (Exception ignored) {}
        try {
            Bundle b = SslCertificate.saveState(cert);
            byte[] der = b.getByteArray("x509-certificate");
            if (der == null) return null;
            return (X509Certificate) java.security.cert.CertificateFactory.getInstance("X.509")
                    .generateCertificate(new java.io.ByteArrayInputStream(der));
        } catch (Exception e) {
            return null;
        }
    }

    private void hideSystemUi() {
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    @Override
    protected void onResume() { super.onResume(); hideSystemUi(); if (web != null) web.reload(); }

    @Override
    protected void onDestroy() { if (web != null) web.destroy(); super.onDestroy(); }
}
