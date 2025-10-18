import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, MessageCircle, Shield, Heart } from 'lucide-react';
import logo from '@/assets/logo.png';

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/10">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center space-y-8 mb-16">
          <div className="flex justify-center">
            <img src={logo} alt="AfyaJamii Logo" className="h-32 w-32" />
          </div>
          <div className="space-y-4">
            <h1 className="text-5xl font-bold text-foreground">AfyaJamii</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              AI-Powered Maternal Health & Nutrition Platform
            </p>
            <p className="text-base text-muted-foreground max-w-xl mx-auto">
              Your trusted companion for pregnancy and postnatal care, providing personalized health monitoring and expert AI guidance
            </p>
          </div>
          <div className="flex gap-4 justify-center">
            <Button asChild size="lg">
              <Link to="/signup">Get Started</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link to="/login">Sign In</Link>
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          <Card>
            <CardHeader>
              <Activity className="h-10 w-10 text-primary mb-2" />
              <CardTitle>Vitals Monitoring</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Track your health metrics with AI-powered risk assessment and instant feedback
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <MessageCircle className="h-10 w-10 text-primary mb-2" />
              <CardTitle>AI Health Chat</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Get personalized nutrition and health advice from our AI assistant anytime
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Shield className="h-10 w-10 text-primary mb-2" />
              <CardTitle>Risk Detection</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Early identification of health concerns with machine learning analysis
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Heart className="h-10 w-10 text-primary mb-2" />
              <CardTitle>Personalized Care</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Tailored recommendations based on your unique pregnancy or postnatal journey
              </CardDescription>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;
